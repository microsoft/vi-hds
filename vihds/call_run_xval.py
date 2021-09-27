# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
from __future__ import absolute_import

from vihds.config import Config, Trainer
from vihds.run_xval import run_on_split, create_parser
from vihds.xval import XvalMerge

def execute(args,settings):
    xval_merge = XvalMerge(args, settings)
    for split_idx in range(1, args.folds + 1):
        print("================================================================")
        print("    FOLD %d of %d"%(split_idx, args.folds))
        print("---------------------------")
        data_pair, val_results = run_on_split(args, settings, split=split_idx)
        if val_results is not None:
            xval_merge.add(split_idx, data_pair, val_results)
    print("================================================================")
    if len(xval_merge.elbo) > 0:
        xval_merge.finalize()
        xval_merge.save()
        xval_merge.make_writer()
        xval_merge.make_images()
        xval_merge.close_writer()
        xval_merge.mark_completed(args.experiment)
        print('Completed')
    else:
        print("No results in xval. Exiting...")

def main():
    parser = create_parser(False)
    args = parser.parse_args()
    settings = Config(args)
    settings.trainer = Trainer(args, add_timestamp=True)
    execute(args,settings)    

if __name__ == "__main__":
    main()