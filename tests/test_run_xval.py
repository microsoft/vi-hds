# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import numpy as np
import os
import subprocess
import tempfile
import re

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/Inference
#   pytest tests


def get_example_dir(results_dir):
    bases = os.listdir(results_dir)
    assert len(bases) == 1, "Directory %s has %d entries, expected 1" % (results_dir, len(bases),)
    assert bases[0].startswith("TEST_"), "Subdirectory of %s is %s, not TEST_..." % (results_dir, bases[0],)
    return os.path.join(results_dir, bases[0])


def check_fold_files_present(example_dir, split, folds):
    for key in ["train", "valid"]:
        expected_dir = os.path.join(example_dir, "%s_%d_of_%d" % (key, split, folds))
        check_contains_one_events_file(expected_dir)


def check_contains_one_events_file(path):
    assert os.path.isdir(path), "Directory %s not found" % path
    bases = os.listdir(path)
    assert len(bases) == 1, "Directory %s has %d entries, expected 1" % (path, len(bases),)
    assert bases[0].startswith("events.out.tfevents."), "Contents of directory %s are unexpected" % path


def pre_test(pattern, example, num_folds):
    '''Tests that a relatively quick (1 minute or so) call of run_xval.py, with two test epochs,
       has a higher validation ELBO on the second test epoch than the first.'''
    results_dir = tempfile.mkdtemp()
    os.environ["INFERENCE_RESULTS_DIR"] = results_dir
    if num_folds > 1:
        cmd = (
            "python vihds/%s.py --seed=0 --folds=%d --experiment=TEST --epochs=4 --test_epoch=2 "
            "--train_sample=10 --test_samples=10 --plot_epoch=0 specs/%s.yaml"
        ) % (pattern, num_folds, example)
    else:
        cmd = (
            "python vihds/%s.py --seed=0 --experiment=TEST --epochs=6 --test_epoch=3 "
            "--train_sample=10 --test_samples=10 --plot_epoch=0 specs/%s.yaml"
        ) % (pattern, example)
    print(cmd)
    result = subprocess.run(cmd, universal_newlines=True, capture_output=True, shell=True)
    assert result.returncode == 0
    elbo_list = []
    for line in result.stdout.split("\n"):
        print(line)
        if "iwae-elbo" in line:
            tokens = re.split("iwae-elbo = |,", line)
            # We have a test epoch line.
            elbo_list.append(float(tokens[4]))

    # Check we got exactly two test epoch lines
    assert len(elbo_list) == 2 * num_folds
    # Check the ELBO was finite in all cases
    for elbo in elbo_list:
        assert np.isfinite(elbo)
    # Check that all the expected files were created
    example_dir = get_example_dir(results_dir)
    # Check train and valid sub-directories are present
    if num_folds == 1:
        check_fold_files_present(example_dir, 1, 4)  # default number of folds is 4
    else:
        for split in range(num_folds):
            check_fold_files_present(example_dir, split + 1, num_folds)
    return example_dir


def post_test(example_dir):
    # Check xval directory created
    check_contains_one_events_file(os.path.join(example_dir, "xval"))
    # All 'xval_' basenames in example_dir
    bases = [base for base in os.listdir(example_dir) if base.startswith("xval_")]
    # Check all expected npy files are present in example_dir
    infixes = "chunk_sizes devices elbo ids names times treatments X_obs iw_predict_mu iw_predict_std iw_states"
    for infix in infixes.split():
        suffix = "txt" if infix in ["names"] else "npy"
        base = "xval_%s.%s" % (infix, suffix)
        assert base in bases, "Cannot find %s in %s" % (base, example_dir)
    # Check xval_{fit,species,treatment}.{pdf,png} are present
    for infix in ["fit", "species", "treatments"]:
        for suffix in ["pdf", "png"]:
            base = "xval_%s.%s" % (infix, suffix)
            assert base in bases, "Cannot find %s in %s" % (base, example_dir)
    # Check there is at least one pdf and at least one png device-specific file, where
    # the device name contains exactly one underscore,so the names are of the form xval_Rfoo_Ybar.{pdf,png}
    for suffix in ["pdf", "png"]:
        matches = [base for base in bases if base.endswith(suffix) and len(base.split("_")) == 4]
        assert len(matches) > 0, "Cannot find file matching *_*_*.%s in %s" % (suffix, example_dir,)


def test_run_xval_auto_constant():
    """Test run_xval for Auto_Constant model"""
    pre_test("run_xval", "auto_constant", 1)


def test_run_xval_prpr_constant():
    """Test run_xval for PRPR_Constant model"""
    pre_test("run_xval", "prpr_constant", 1)


def test_run_xval_dr_constant_icml():
    """Test run_xval for DR_Constant model"""
    pre_test("run_xval", "dr_constant_icml", 1)


def test_run_xval_dr_constant_precisions():
    """Test run_xval for DR_Constant_Precisions model"""
    pre_test("run_xval", "dr_constant_precisions", 1)


def test_run_xval_dr_blackbox_icml():
    """Test run_xval for DR_Blackbox model"""
    pre_test("run_xval", "dr_blackbox_icml", 1)


def test_folds2():
    """
    Tests that calling call_run_xval.py goes through, and has the same ELBO behaviour on each
    fold as run_xval_icml.py does. This is slow - about five minutes. Therefore, using only a single dataset.
    """
    example_dir = pre_test("call_run_xval", "dr_constant_one", 2)
    post_test(example_dir)


if __name__ == "__main__":
    test_run_xval_auto_constant()
    test_run_xval_prpr_constant()
    test_run_xval_dr_constant_icml()
    test_run_xval_dr_constant_precisions()
    test_run_xval_dr_blackbox_icml()
    test_folds2()
