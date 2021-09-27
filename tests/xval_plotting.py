# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import time
import pandas as pd
from collections import OrderedDict

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/vi-hds
#   pytest tests

from vihds.config import Config, locate_yml, Trainer
from vihds.run_xval import create_parser
from vihds.xval import XvalMerge
from vihds import plotting
from vihds.utils import Results
from vihds.datasets import build_datasets

import cProfile
import pstats
import io


def main(res):
    device_ids = list(range(len(res.settings.devices)))

    # print("Making summary figure:")
    # f_summary = plotting.plot_prediction_summary(res.settings.devices, res.species_names, res.times,
    #     res.X_obs, res.iw_predict_mu, res.iw_predict_std, res.devices, '-')
    # res.save_figs(f_summary,'xval_fit')

    # if res.settings.separate_conditions is True:
    #     print("Making treatment figure")
    #     f_treatments = plotting.xval_treatments(res, device_ids)
    #     res.save_figs(f_treatments,'xval_treatments')

    print("Making species figure")
    f_species = plotting.species_summary(
        res.species_names,
        res.treatments,
        res.devices,
        res.times,
        res.iw_states,
        device_ids,
        res.settings,
        normalise=True,
    )
    res.save_figs(f_species, "xval_species")

    # print("Making global parameters figure")
    # f_gparas = plotting.xval_global_parameters(res)
    # if f_gparas is not None:
    #     res.save_figs(f_gparas,'xval_global_parameters')

    # print("Making variable parameters figure")
    # f_vparas = plotting.xval_variable_parameters(res)
    # if f_vparas is not None:
    #     res.save_figs(f_vparas,'xval_variable_parameters')

    # print("Making summary device figures")
    # for u in device_ids:
    #     print("- %s" % res.settings.pretty_devices[u])
    #     device = res.settings.devices[u]
    #     f_summary_i = plotting.xval_fit_summary(res, u, separatedInputs=res.settings.separate_conditions)
    #     res.save_figs(f_summary_i, 'xval_summary_%s' % device)

    # print("Making individual device figures")
    # for u in device_ids:
    #     print("- %s" % res.settings.pretty_devices[u])
    #     device = res.settings.devices[u]
    #     f_indiv1 = plotting.xval_individual(res, u)
    #     res.save_figs(f_indiv1, 'xval_indiv1_%s' % device)
    #     f_indiv2 = plotting.xval_individual_2treatments(res, u)
    #     res.save_figs(f_indiv2, 'xval_indiv2_%s' % device)

    # res.make_writer()
    # res.make_images()


def load_xval(log_dir):
    # Load a spec (YAML)
    parser = create_parser(True)
    log_dir = ".\\.vihds_cache"
    yaml = locate_yml(log_dir)
    args = parser.parse_args([yaml])
    settings = Config(args)
    settings.trainer = Trainer(args, log_dir=log_dir)
    res = XvalMerge(args, settings)
    res.load()
    return res


def load_cache(yaml):
    parser = create_parser(True)
    args = parser.parse_args([yaml])
    settings = Config(args)
    data_pair = build_datasets(args, settings, settings.data.load)
    settings.trainer = Trainer(args, log_dir=".")
    res = Results()
    res.load()
    res.elbo_list = [res.elbo]
    xval = XvalMerge(args, settings)
    xval.add(1, data_pair, res)
    xval.finalize()
    return xval


if __name__ == "__main__":

    # log_dir = '.\\results\\dr_icml'
    # res = load_xval(log_dir)
    res = load_cache("specs/dr_constant_precisions.yaml")

    pr = cProfile.Profile()
    pr.enable()

    main(res)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()

    with open("profile_plotting.txt", "w") as f:
        f.write(s.getvalue())
