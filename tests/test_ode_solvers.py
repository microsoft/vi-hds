# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# pylint: disable=not-callable,no-member

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/vi-hds
#   pytest tests

from vihds.datasets import build_datasets
from vihds.config import Config
from vihds.parameters import Parameters
from vihds.training import batch_to_device, Training
from vihds.vae import build_model
from vihds.run_xval import create_parser

# Setup config, data, parameters and model
parser = create_parser(True)
samples = 5
args = parser.parse_args(["--train_samples=%d" % samples, "--test_samples=%d" % samples, "specs/dr_constant_one.yaml",])


def simulate(settings, model, theta, batch, solver, adjoint, verbose):
    settings.params.solver = solver
    test_start = time.time()
    sol = model.decoder.ode_model.simulate(
        settings, batch.times, theta, batch.inputs, batch.dev_1hot, condition_on_device=False,
    )
    test_time = time.time() - test_start
    if verbose:
        print("\n- %s: %1.3f seconds" % (solver, test_time), end="")
    else:
        print(".", end="")
    return sol[:, :, :, -1].cpu().detach().numpy(), test_time


def run(args, verbose=False):
    settings = Config(args)
    data = build_datasets(args, settings)
    parameters = Parameters(settings.params)
    model = build_model(args, settings, data, parameters)
    training = Training(args, settings, data, parameters, model)

    # Prepare a data sample
    train_loader = DataLoader(dataset=data.train, batch_size=settings.params.n_batch, shuffle=True)
    batch = next(iter(train_loader))
    batch = batch_to_device(data.train.dataset.times, settings.device, batch)

    # Run the encoder to generate q, then sample some thetas
    q = training.model.encoder(batch)
    u = training.model.sample_u(len(batch.inputs), samples)
    theta = q.sample(u, training.model.device)
    # clipped_theta = model.encoder.p.clip(theta, stddevs=4)

    # Define simulation variables and run simulator
    solvers = [
        "modeuler",
        "modeulerwhile",
        "dopri5",
        "dopri8",
        "midpoint",
        "rk4",
    ]  # ,'adaptive_heun','bosh3']
    print("--------------------")
    # Run
    print("Direct versions of the solvers ", end="")
    dir_solutions, dir_times = zip(
        *[simulate(settings, training.model, theta, batch, solver, False, verbose) for solver in solvers]
    )
    print("\nAdjoint versions of the solvers ", end="")
    settings.params.adjoint_solver = True
    adj_solutions, adj_times = zip(
        *[simulate(settings, training.model, theta, batch, solver, True, verbose) for solver in solvers[2:]]
    )

    sol_array = np.array(dir_solutions + adj_solutions)
    std = np.std(sol_array, axis=0)
    mean = np.mean(sol_array, axis=0)
    cvs = std / mean
    cv_max = np.max(cvs)
    print("\nMaximum CV: %1.3f" % cv_max)
    print("--------------------")
    assert cv_max < 0.05, "Coefficient of variation across solvers should not exceed 5%"
    df = pd.DataFrame.from_dict({"Direct": dir_times, "Adjoint": [float("nan")] * 2 + list(adj_times)})
    df.index = solvers
    print("Times")
    print(df)


# Run in CPU mode
def test_solvers_cpu():
    print("\nTesting solvers with CPU")
    print("--------------------------")
    run(args)


# Run in GPU mode
def test_solvers_gpu():
    print("\nTesting solvers with GPU")
    print("--------------------------")
    args.gpu = 0
    run(args)


if __name__ == "__main__":
    # test_solvers_cpu()
    test_solvers_gpu()
