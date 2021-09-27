# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
from __future__ import absolute_import
import argparse

# Local imports
from vihds.config import Config, Trainer
from vihds.datasets import build_datasets
from vihds.parameters import Parameters
from vihds.training import Training
from vihds.xval import XvalMerge
from vihds.vae import build_model

def create_parser(with_split: bool):
    parser = argparse.ArgumentParser(description='VI-HDS')
    parser.add_argument('yaml', type=str, help='Name of yaml spec file')
    parser.add_argument('--experiment', type=str, default='unnamed', help='Name for experiment, also location of tensorboard and saved results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--test_epoch', type=int, default=20, help='Frequency of calling test')
    parser.add_argument('--plot_epoch', type=int, default=100, help='Frequency of plotting figures')
    parser.add_argument('--train_samples', type=int, default=200, help='Number of samples from q, per datapoint, during training')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of samples from q, per datapoint, during testing')
    parser.add_argument('--dreg', type=bool, default=True, help='Use DReG estimator')
    parser.add_argument('--precision_hidden_layers', type=int, default=None, help='Number of hidden layers to use in neural precisions')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print more information about parameter setup')
    parser.add_argument('--gpu', type=int, default=None, help='Use GPU device (default None is CPU mode')
    if with_split:
        # We make --heldout (heldout device) and --split mutually exclusive. Really we'd like to say it's allowed
        # to specify *either* --heldout *or* --split and/or --folds, but argparse isn't expressive enough for that.
        # So if the user specifies --heldout and --folds, there won't be a complaint here, but --folds will be ignored.
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--heldout', type=str, help='name of held-out device, e.g. R33S32_Y81C76')
        group.add_argument('--split', type=int, default=1, help='Specify split in 1:folds for cross-validation')
        group.add_argument('--figures', action='store_true', default=False, help='Create figures (default: False)')
    parser.add_argument('--folds', type=int, default=4, help='Cross-validation folds')
    return parser

def run_on_split(args, settings, split=None):
    '''Run one train-test split'''
    if getattr(args, 'heldout', None):
        print("Heldout device is %s" % args.heldout)
    else:
        args.heldout = None # in case not defined at all
        if split is not None:
            args.split = split
    data = build_datasets(args, settings)
    parameters = Parameters(settings.params)
    model = build_model(args, settings, data, parameters)
    training = Training(args, settings, data, parameters, model)
    return data, training.run()

def main():
    parser = create_parser(True)
    args = parser.parse_args()
    settings = Config(args)
    settings.trainer = Trainer(args, add_timestamp=True)
    data_pair, val_results = run_on_split(args, settings)

    if (val_results is not None) and settings.trainer is not None:
        xval_merge = XvalMerge(args, settings)
        xval_merge.add(1, data_pair, val_results)
        xval_merge.finalize()
        xval_merge.save()
        xval_merge.mark_completed(args.experiment)
        if args.figures:
            xval_merge.make_writer()
            xval_merge.make_images()
            xval_merge.close_writer()

if __name__ == "__main__":
    main()