# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import absolute_import

from data import procdata
from simulation.run_xval_icml import run_on_split, create_parser
from simulation.xval import XvalMerge
from utils import Trainer

def main():
    parser = create_parser(False)
    args = parser.parse_args()
    print(args)
    trainer = Trainer(args, args.yaml, add_timestamp=True)
    xval_merge = XvalMerge(args, trainer)
    for split_idx in range(1, args.folds + 1):
        print("---------------------------------------------------------------------------")
        print("    FOLD %d of %d"%(split_idx, args.folds))
        print("---------------------------------------------------------------------------")
        data_pair, val_results = run_on_split(args, split_idx, xval_merge.trainer)
        xval_merge.add(split_idx, data_pair, val_results)
    xval_merge.finalize()
    xval_merge.make_writer(xval_merge.trainer.tb_log_dir)
    xval_merge.prepare_treatment()
    xval_merge.make_images(procdata.ProcData())
    xval_merge.close_writer()
    xval_merge.save(xval_merge.trainer.tb_log_dir)
    #xval_merge.load(xval_merge.trainer.tb_log_dir)
    print('Completed')

if __name__ == "__main__":
    main()