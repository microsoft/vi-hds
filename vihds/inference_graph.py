# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import yaml
import munch
import vihds.run_xval as rxval

class Edge():
    def __init__(self,source, sourceParam,target, targetParam):
        self.source = source
        self.sourceParam = sourceParam
        self.target = target
        self.targetParam = targetParam        


def process_node_args(name,yamlargs,graph_name):
    split = False
    argarr = []
    ## Should this also see if any one of those 3 arguments are in the command line?
    if 'split' in yamlargs:
        True
        argarr.append("--split="+str(yamlargs.split))

    if 'spec' in yamlargs:
        argarr.append(yamlargs.spec)
    else:
        raise ValueError('Node ' + name + ' missing spec property')
    
    if 'experiment' in yamlargs: 
        argarr.append('--experiment=' + graph_name + '/' + yamlargs.experiment)
    else:
        raise ValueError('Node ' + name + ' missing experiment property')
    
    if 'seed' in yamlargs: 
        argarr.append('--seed=' + str(yamlargs.seed))
    
    if 'train_samples' in yamlargs: 
        argarr.append('--train_samples=' + str(yamlargs.train_samples))
    
    if 'test_samples' in yamlargs: 
        argarr.append('--test_samples=' + str(yamlargs.test_samples))
    
    if 'epochs' in yamlargs: 
        argarr.append('--epochs=' + str(yamlargs.epochs))
    
    if 'test_epoch' in yamlargs: 
        argarr.append('--test_epoch=' + str(yamlargs.test_epoch))
    
    if 'plot_epoch' in yamlargs: 
        argarr.append('--plot_epoch=' + str(yamlargs.plot_epoch))
    
    if 'gpu' in yamlargs:
        argarr.append('--gpu=' + str(yamlargs.gpu))

    ### Should probably add other arguments as well...


    parser = rxval.create_parser(split)
    args = parser.parse_args(argarr)
    
    return args

class Node():
    def __init__(self,name,yamlargs,graph_name):
        self.name = name
        self.stage = None
        self.incoming = []
        self.outgoing = []
        self.args = process_node_args(name,yamlargs,graph_name)
    def addIncomingEdge(self,edge):
        self.incoming.append(edge)
    def addOutgoingEdge(self,edge):
        self.outgoing.append(edge)
    def setStage(self,stage):
        self.stage = stage

def setStage(node):    
    if node.stage is None: 
        if not node.incoming: 
            node.setStage(0)
        else:
            stage = 0
            for incoming in node.incoming: 
                if incoming.source.stage is None: 
                    setStage(incoming.source)
                if incoming.source.stage > stage:
                    stage = incoming.source.stage
            node.setStage(stage + 1)        
    else:
        pass

def create_inference_graph(graphyml,graph_name):
    with open(graphyml,'r') as f:
        graph = munch.munchify(yaml.safe_load(f))
    nodemap = {}
    for key in graph.nodes.keys():
        nodemap[key] = Node(key,graph.nodes[key],graph_name)
    for edge in graph.edges:
        source = nodemap[edge['from'].node]
        target = nodemap[edge['to'].node]
        e = Edge(source,edge['from'].parameter,target,edge['to'].parameter)
        source.addOutgoingEdge(e)
        target.addIncomingEdge(e)
    for node in nodemap.values():
        setStage(node)
    return nodemap


## Returns a map of the nodes where the key is "Level or stage" and the value is the list of nodes that can be executed in parallel at that stage. 
def arrange_by_stage(nodes):
    stagemap = {}
    for node in nodes: 
        if node.stage not in stagemap:
            stagemap[node.stage] = []           
        stagemap[node.stage].append(node)
    return stagemap
