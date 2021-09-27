# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import tempfile
import torch

# pylint: disable=no-member

from vihds.datasets import build_datasets
from vihds.run_xval import create_parser
from vihds.parameters import Parameters
from vihds.vae import build_model
from vihds.config import Config, Trainer
from vihds.training import Training, batch_to_device


def run(yml):
    results_dir = tempfile.mkdtemp()
    os.environ["INFERENCE_RESULTS_DIR"] = results_dir

    samples = 20
    parser = create_parser(True)
    args = parser.parse_args(
        [
            "--train_samples=%d" % samples,
            "--test_samples=%d" % samples,
            "--test_epoch=5",
            "--plot_epoch=0",
            "--epochs=5",
            "--seed=0",
            yml,
        ]
    )
    settings = Config(args)
    settings.trainer = Trainer(args, add_timestamp=True)
    data = build_datasets(args, settings)
    parameters = Parameters(settings.params)
    model = build_model(args, settings, data, parameters)
    training = Training(args, settings, data, parameters, model)

    training.model.train()
    # Evaluate the encoder to produce a q
    batch = training.train_data
    batch = batch_to_device(data.train.dataset.times, settings.device, batch)
    batch_results, theta, q, p = training.model(batch, args.train_samples)

    elbo = training.cost(batch, batch_results, theta, q, p).elbo
    elbo.backward()

    nans = []
    for name, dist in q.distributions.items():
        for pname in dist.param_names:
            grad = getattr(dist, pname).grad
            if grad is not None:
                isnan = torch.isnan(grad)
                if isnan.any():
                    nans.append("%s.%s" % (name, pname))
    assert len(nans) == 0, "NaN gradients for %s" % (", ".join(nans))


def test_grad_auto():
    yml = "specs/auto_constant.yaml"
    run(yml)


def test_grad_auto_prec():
    yml = "specs/auto_constant_precisions.yaml"
    run(yml)


def test_grad_prpr():
    yml = "specs/prpr_constant.yaml"
    run(yml)


def test_grad_prpr_prec():
    yml = "specs/prpr_constant_precisions.yaml"
    run(yml)


def test_grad_dr_one():
    yml = "specs/dr_constant_one.yaml"
    run(yml)


def test_grad_dr_icml():
    yml = "specs/dr_constant_icml.yaml"
    run(yml)


def test_grad_dr_blackbox():
    yml = "specs/dr_blackbox_icml.yaml"
    run(yml)


def test_grad_dr_v2():
    yml = "specs/dr_constant_v2.yaml"
    run(yml)


def test_grad_dr_precisions():
    yml = "specs/dr_constant_precisions.yaml"
    run(yml)


def test_grad_dr_precisions_v2():
    yml = "specs/dr_constant_precisions_v2.yaml"
    run(yml)


if __name__ == "__main__":
    # test_grad_auto()
    test_grad_auto_prec()
    test_grad_prpr()
    test_grad_prpr_prec()
    test_grad_dr_one()
    test_grad_dr_icml()
    test_grad_dr_blackbox()
    test_grad_dr_v2()
    test_grad_dr_precisions()
    test_grad_dr_precisions_v2()
