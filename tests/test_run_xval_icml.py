# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import os
import subprocess
import sys
import tempfile
import re

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/Inference
#   pytest tests

def get_example_dir(results_dir):
    bases = os.listdir(results_dir)
    assert len(bases) == 1, 'Directory %s has %d entries, expected 1' % (results_dir, len(bases))
    assert bases[0].startswith('TEST_'), 'Subdirectory of %s is %s, not TEST_...' % (results_dir, bases[0])
    return os.path.join(results_dir, bases[0])

def check_fold_files_present(example_dir, split, folds):
    for key in ['train', 'valid']:
        expected_dir = os.path.join(example_dir, '%s_%d_of_%d' % (key, split, folds))
        check_contains_one_events_file(expected_dir)

def check_contains_one_events_file(path):
    assert os.path.isdir(path), 'Directory %s not found' % path
    bases = os.listdir(path)
    assert len(bases) == 1, 'Directory %s has %d entries, expected 1' % (path, len(bases))
    assert bases[0].startswith('events.out.tfevents.'), 'Contents of directory %s are unexpected' % path


def pre_test(pattern, num_folds):
    # Form of subprocess.run below is not guaranteed to work on Python 2.
    assert sys.version_info[0] == 3, 'This test will only run on Python 3'
    results_dir = tempfile.mkdtemp()
    os.environ['INFERENCE_RESULTS_DIR'] = results_dir
    if num_folds > 1:
        cmd = ('python src/%s.py --yaml=specs/dr_blackbox_xval.yaml --folds=%d '
            '--experiment=TEST --epochs=8 --test_epoch=4 --train_sample=10 --test_samples=10') % (pattern, num_folds)
    else:
        cmd = ('python src/%s.py --yaml=specs/dr_blackbox_xval.yaml '
            '--experiment=TEST --epochs=8 --test_epoch=4 --train_sample=10 --test_samples=10') % pattern
    cmd_tokens = cmd.split()
    result = subprocess.run(cmd_tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0
    elbo_list = []
    for line in result.stdout.split('\n'):
        print(line)
        if "iwae-elbo" in line:
            tokens = re.split("iwae-elbo = |,", line)
            # We have a test epoch line.
            elbo_list.append(float(tokens[4]))

    # Check we got exactly two test epoch lines
    assert len(elbo_list) == 2 * num_folds
    # Check the ELBO increased between the two test epochs.
    for i in range(num_folds):
        assert elbo_list[2*i] < elbo_list[2*i+1]
    # Check that all the expected files were created
    example_dir = get_example_dir(results_dir)
    # Check train and valid sub-directories are present
    if num_folds==1:
        check_fold_files_present(example_dir, 1, 4)    # default number of folds is 4
    else:
        for split in range(num_folds):
            check_fold_files_present(example_dir, split + 1, num_folds)
    return example_dir

def post_test(example_dir):
    # Check xval directory created
    check_contains_one_events_file(os.path.join(example_dir, 'xval'))
    # All 'xval_' basenames in example_dir
    bases = [base for base in os.listdir(example_dir) if base.startswith('xval_')]
    # Check all expected npy files are present in example_dir
    infixes = 'chunk_sizes devices elbo ids log_normalized_iws names precisions times treatments X_obs X_post_sample X_sample'
    for infix in infixes.split():
        suffix = 'txt' if infix in ['names'] else 'npy'
        base = 'xval_result_%s.%s' % (infix, suffix)
        assert base in bases, 'Cannot find %s in %s' % (base, example_dir)
    # Check xval_{fit,species,treatment}.{pdf,png} are present
    for infix in ['fit', 'species', 'treatments']:
        for suffix in ['pdf', 'png']:
            base = 'xval_%s.%s' % (infix, suffix)
            assert base in bases, 'Cannot find %s in %s' % (base, example_dir)
    # Check there is at least one pdf and at least one png device-specific file, where
    # the device name contains exactly one underscore,so the names are of the form xval_Rfoo_Ybar.{pdf,png}
    for suffix in ['pdf', 'png']:
        matches = [base for base in bases if base.endswith(suffix) and len(base.split('_')) == 4]
        assert len(matches) > 0, 'Cannot find file matching *_*_*.%s in %s' % (suffix, example_dir)


def test_run_xval_icml():
    '''
    Tests that a relatively quick (1 minute or so) call of run_xval_icml.py, with two test epochs,
    has a higher validation ELBO on the second test epoch than the first.
    '''
    pre_test('run_xval_icml', 1)

def test_folds2():
    '''
    Tests that calling call_run_xval.py goes through, and has the same ELBO behaviour on each
    fold as run_xval_icml.py does. This is slow - about five minutes.
    '''
    example_dir = pre_test('call_run_xval', 2)
    post_test(example_dir)