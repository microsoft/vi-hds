# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
import tempfile

# Call tests in this file by running "pytest" on the directory containing it. For example:
#   cd ~/Inference/VariationalPyCRN
#   pytest PyCRN/tests

CALL_FORMAT = ('python3 PyCRN/simulation/%s.py --yaml=PyCRN/specs/dr_constant_xval.yaml '
               '--experiment=EXAMPLE --epochs 8 --test_epoch 4 --test_samples 100')

def test_run_xval_icml():
    '''
    Tests that a relatively quick (1m20s or so) call of run_xval_icml.py, with two test epochs,
    has a higher validation ELBO on the second test epoch than the first.
    '''
     # Form of subprocess.run below is not guaranteed to work on Python 2.
    assert sys.version_info[0] == 3, 'This test will only run on Python 3'
    # Change to VariationalPyCRN directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_dir = tempfile.mkdtemp()
    os.environ['INFERENCE_RESULTS_DIR'] = results_dir
    cmd = CALL_FORMAT % 'run_xval_icml'
    cmd_tokens = cmd.split()
    result = subprocess.run(cmd_tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0
    elbo_list = []
    for line in result.stdout.split('\n'):
        tokens = line.strip().split()
        if len(tokens) >= 14 and tokens[0] in ['Split', 'Heldout']:
            # We have a test epoch line. The validation ELBO is the 14th token.
            elbo_list.append(float(tokens[13]))
    # Check we got exactly two test epoch lines
    assert len(elbo_list) == 2
    # Check the ELBO increased between the two test epochs.
    assert elbo_list[0] < elbo_list[1]
    # Check that all the expected files were created
    example_dir = get_example_dir(results_dir)
    # Check train_1_4 and valid_1_4 are present
    check_fold_files_present(example_dir, 1, 4)

def get_example_dir(results_dir):
    bases = os.listdir(results_dir)
    assert len(bases) == 1, 'Directory %s has %d entries, expected 1' % (results_dir, len(bases))
    assert bases[0].startswith('EXAMPLE_'), 'Subdirectory of %s is %s, not EXAMPLE_...' % (results_dir, bases[0])
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

def test_call_run_xval():
    '''
    Tests that calling call_run_xval.py goes through, and has the same ELBO behaviour on each
    fold as run_xval_icml.py does. This is slow - about ten minutes.
    '''
     # Form of subprocess.run below is not guaranteed to work on Python 2.
    assert sys.version_info[0] == 3
    # Change to VariationalPyCRN directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_dir = tempfile.mkdtemp()
    os.environ['INFERENCE_RESULTS_DIR'] = results_dir
    cmd = CALL_FORMAT % 'call_run_xval'
    cmd_tokens = cmd.split()
    print(cmd_tokens)
    result = subprocess.run(cmd_tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert result.returncode == 0
    elbo_list = []
    for line in result.stdout.split('\n'):
        tokens = line.strip().split()
        if len(tokens) >= 14 and tokens[0] == 'Split':
            # We have a test epoch line. The validation ELBO is the 14th token.
            elbo_list.append(float(tokens[13]))
    # Check we got exactly 2 (epochs) x 4 (folds) = 8 test epoch lines
    assert len(elbo_list) == 8
    # Check the ELBO increased between the two test epochs in each fold
    for i in range(0, 8, 2):
       assert elbo_list[i] < elbo_list[i+1]
    # Check train_ and valid_ directories look OK
    example_dir = get_example_dir(results_dir)
    for split in range(4):
        check_fold_files_present(example_dir, split + 1, 4)
    # Check xval directory created
    check_contains_one_events_file(os.path.join(example_dir, 'xval'))
    # All 'xval_' basenames in example_dir
    bases = [base for base in os.listdir(example_dir) if base.startswith('xval_')]
    # Check all expected npy files are present in example_dir
    infixes = 'chunk_sizes devices elbo ids log_normalized_iws names precisions times treatments X_obs X_post_sample X_sample'
    for infix in infixes.split():
        suffix = 'txt' if infix in ['elbo', 'names'] else 'npy'
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
        matches = [base for base in bases if base.endswith(suffix) and len(base.split('_')) == 3]
        assert len(matches) > 0, 'Cannot find file matching *_*_*.%s in %s' % (suffix, example_dir)