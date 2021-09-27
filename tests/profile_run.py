# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import tempfile
import cProfile, pstats, io
from pstats import SortKey

import torch
from torch.utils.data import DataLoader
from vihds.datasets import build_datasets
from vihds.run_xval import create_parser, run_on_split
from vihds.config import Config, locate_yml, Trainer

if __name__ == '__main__':

    results_dir = tempfile.mkdtemp()
    os.environ['INFERENCE_RESULTS_DIR'] = results_dir

    #yml = 'specs\\dr_constant_one.yaml'    # Single files
    #yml = 'specs\\dr_constant_icml.yaml'    # Multiple files
    #yml = 'specs\\dr_constant_precisions.yaml'    # Multiple files
    yml = 'specs\\dr_blackbox_icml.yaml'    # Multiple files
    samples = 20
    epochs = 5
    parser = create_parser(True)
    args = parser.parse_args([
        "--train_samples=%d"%samples, "--test_samples=%d"%samples, 
        "--test_epoch=5", "--plot_epoch=0", 
        "--epochs=5", "--seed=0", yml])
    settings = Config(args)
    settings.trainer = Trainer(args, add_timestamp=True)
    
    pr = cProfile.Profile()
    pr.enable()
    data_pair, val_results = run_on_split(args, settings)
    pr.disable()
    
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())
    with open('profile_stats.txt', 'w') as f:
        f.write(s.getvalue())