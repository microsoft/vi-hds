# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
from torch.utils.data import DataLoader

# pylint: disable=no-member

from vihds.datasets import build_datasets
from vihds.run_xval import create_parser
from vihds.config import Config
from vihds.parameters import Parameters
from vihds.training import batch_to_device
from vihds.vae import build_model


def test_shapes():
    # yml = 'specs/dr_constant_one.yaml'    # Single files
    yml = "specs/dr_constant_icml.yaml"  # Multiple files
    parser = create_parser(True)
    args = parser.parse_args([yml])
    settings = Config(args)

    data = build_datasets(args, settings)
    parameters = Parameters(settings.params)
    model = build_model(args, settings, data, parameters)

    # Test dataset size
    nf = args.folds
    assert data.n_train == 312 * (nf - 1) / nf, "Training set the correct size"
    assert data.n_test == 312 / nf, "Test set the correct size"

    # Test batch loader
    n_batch = 36
    train_loader = DataLoader(dataset=data.train, batch_size=n_batch, shuffle=True)
    batch = next(iter(train_loader))
    batch = batch_to_device(data.train.dataset.times, settings.device, batch)
    assert batch.devices.shape == torch.Size([n_batch]), "Batch has right shape for 'devices'"
    assert batch.dev_1hot.shape == torch.Size(
        [n_batch, settings.data.device_depth]
    ), "Batch has right shape for 'dev_1hot'"
    assert batch.inputs.shape == torch.Size(
        [n_batch, len(settings.data.conditions)]
    ), "Batch has right shape for 'inputs'"
    assert batch.observations.shape == torch.Size([n_batch, 4, 86]), "Batch has right shape for 'observations'"

    # Test conditional encoder shape
    delta_obs = batch.observations[:, :, 1:] - batch.observations[:, :, :-1]
    q = model.encoder.conditional(delta_obs)
    print("q:", q.shape)

    assert q.shape == (n_batch, settings.params.n_hidden), "Shape of encoder output"


if __name__ == "__main__":
    test_shapes()
