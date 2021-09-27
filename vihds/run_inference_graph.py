# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import os
import argparse
import munch
import numpy as np
import vihds.inference_graph as ig
import vihds.config
import vihds.call_run_xval as CallRunXval
from vihds.config import Config, Trainer


def create_parser():
    parser = argparse.ArgumentParser(description="VI-HDS")
    parser.add_argument("yaml", type=str, help="Name of yaml spec file for the inference graph")
    parser.add_argument(
        "--graph",
        type=str,
        default="unnamed",
        help="Name for the inference graph and also location of tensorboard and saved results of all the nodes in the graph",
    )
    return parser


def pooled_prec(xarr):
    size = len(xarr)
    den = 0
    for x in xarr:
        den = den + (1 / x)
    return size / den


def propagate_params(node, settings, resultmap):
    for incoming in node.incoming:
        print(
            "Incoming node for " + node.name + " is " + incoming.source.name + " with parameter " + incoming.sourceParam
        )
        inresultfp = resultmap[incoming.source.name]
        xvalfp = os.path.join(inresultfp, "xval_q_values.npy")
        labelsfp = os.path.join(inresultfp, "xval_q_names.txt")
        xval = np.load(xvalfp, allow_pickle=True)
        with open(labelsfp) as file:
            xlabels = [line.rstrip() for line in file]

        avgmu = np.mean(xval[xlabels.index(incoming.sourceParam + ".mu")])  ## Average
        prec = pooled_prec(xval[xlabels.index(incoming.sourceParam + ".prec")])
        distribution = "LogNormal"
        keyparams = ["global", "local", "shared"]
        for key in keyparams:
            if incoming.targetParam in settings.params[key]:
                print(
                    "Target Parameter for "
                    + node.name
                    + " is "
                    + incoming.targetParam
                    + " which is in "
                    + key
                    + " in settings.param"
                )
                newdist = {}
                newdist["distribution"] = distribution
                newdist["mu"] = avgmu
                newdist["sigma"] = prec
                settings.params[key][incoming.targetParam] = munch.munchify(newdist)


def save_propagatedParameters(params, folder):
    fp = os.path.join(folder, "propagatedParams.txt")
    with open(fp, "w") as file:
        file.write(str(params))
        file.close()


def run_graph(graph_name, staged_nodes):
    ##Do some preprocessing and checks here?

    ## Create directory for results
    rootpath = os.path.join(vihds.config.get_results_directory(), graph_name)

    resultmap = {}

    if not os.path.exists(rootpath):
        #    print("Path doesn't exist yet. Creating folder: " + rootpath)
        os.makedirs(rootpath)
    # else:
    #     print('Graph result folder exists..')
    subfolders = os.listdir(rootpath)

    for stage in range(len(staged_nodes)):
        print("------------------------------------------------------------------------------------------------------")
        print("Current stage of the graph: " + str(stage) + " which has " + str(len(staged_nodes[stage])) + " node(s).")
        for node in staged_nodes[stage]:
            print("#############################")
            print("Processing Node: " + node.name)
            run_node = True
            for subfolder in subfolders:
                if subfolder.startswith(node.name):
                    sbpath = [f.path for f in os.scandir(rootpath) if (f.is_dir() and (f.name == subfolder))][0]
                    completedpath = os.path.join(sbpath, "completed.txt")
                    if os.path.exists(completedpath):
                        with open(completedpath) as file:
                            completednode = file.read()
                        if completednode == node.args.experiment:
                            resultmap[node.name] = sbpath
                            run_node = not (completednode == node.args.experiment)

            if run_node:
                print("About to start running Node:" + node.name + ".")
                settings = Config(node.args)
                settings.trainer = Trainer(node.args, add_timestamp=True)
                print("-----------------------------")
                print("Before propagating parameters:")
                print(settings.params)
                print("-----------------------------")
                propagate_params(node, settings, resultmap)
                print("After propagating parameters:")
                print(settings.params)
                save_propagatedParameters(settings.params, settings.trainer.tb_log_dir)
                print("-----------------------------")
                CallRunXval.execute(node.args, settings)
                resultmap[node.name] = settings.trainer.tb_log_dir
            else:
                print("Node:" + node.name + " execution completed.")


def main():
    parser = create_parser()
    graphArgs = parser.parse_args()
    if graphArgs.yaml is None:
        print("Please specify an Inference Graph.")
    else:
        graph_map = ig.create_inference_graph(graphArgs.yaml, graphArgs.graph)
        staged_nodes = ig.arrange_by_stage(graph_map.values())
        run_graph(graphArgs.graph, staged_nodes)
        print("------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
