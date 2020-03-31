'''Script to generate MDS plots for a DE trajectory on the chosen benchmark
'''

import os
import sys
import json
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import MDS

import ConfigSpace

from denas import DE

from matplotlib import rcParams
rcParams["font.size"] = "10"
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1


def get_mds(X):
    if X.shape[1] <= 2:
        return X
    embedding = MDS(n_components=2)
    return embedding.fit_transform(X)


def generate_colors(i, budgets=None, colors=0):
    # alphas = np.linspace(0.1, 1, i)
    alphas = np.geomspace(0.1, 1, i)
    rgba_colors = np.zeros((i, 4))
    rgba_colors[:, 3] = alphas
    if budgets is None:
        light, dark = color_pairs(colors)
        rgba_colors[:, 0] = light[0]
        rgba_colors[:, 1] = light[1]
        rgba_colors[:, 2] = light[2]
        return rgba_colors
    budgets = budgets[:i]
    budget_set = np.unique(budgets)
    for j, b in enumerate(budget_set):
        idxs = np.where(budgets == b)[0]
        light, dark = color_pairs(j)
        rgba_colors[idxs, 0] = light[0]
        rgba_colors[idxs, 1] = light[1]
        rgba_colors[idxs, 2] = light[2]
    return rgba_colors


def color_pairs(i=0):
    colors = [((0.60784314, 0.54901961, 0.83137255),  #9B8CD4 -- BLUE BELL
               (0.18039216, 0.21960784, 0.18039216)), #2E382E -- JET
              ((0.83921569, 0.70196078, 0.02352941),  #D6B306 -- VIVID AMBER
               (0.83921569, 0.34901961, 0.00000000)), #D65900 -- TENNE
              ((0.56470588, 0.80784314, 0.45098039),  #90CE73 -- PISTACHIO
               (0.27843137, 0.60784314, 0.49803922)), #479B7F -- WINTERGREEN DREAM
              ((0.75294118, 0.37647059, 0.36078431),  #C0605C -- INDIAN RED
               (0.43137255, 0.01176471, 0.12941176)), #6E0321 -- BURGUNDY
              ((0.45882353, 0.29803922, 0.16078431),  #754C29 -- DONKEY BROWN
               (0.15294118, 0.09019608, 0.05490196)), #27170E -- ZINNWALDITE BROWN
              ((0.96862745, 0.72156863, 0.00392157),  #F7B801 -- SELECTIVE YELLOW
               (0.65098039, 0.00000000, 0.00000000))  #A60000 -- DARK CANDY APPLE RED
    ]
    return colors[i]


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def get_configuration_space(max_nodes, search_space):
    cs = ConfigSpace.ConfigurationSpace()
    for i in range(1, max_nodes):
        for j in range(i):
            node_str = '{:}<-{:}'.format(i, j)
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
    return cs

# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def config2structure_func(max_nodes):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return CellStructure( genotypes )
    return config2structure


parser = argparse.ArgumentParser()

parser.add_argument('--benchmark', default='101', type=str, nargs='?',
                    choices=['101', '1shot1', '201'], help='select benchmark')
parser.add_argument('--benchmark_type', default=None, type=str, nargs='?')
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='number of DE generations')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the plot will be saved')
parser.add_argument('--name', default="None", type=str,
                    help='file name for the PNG plot to be saved')
parser.add_argument('--title', default="None", type=str,
                    help='title for the plot')
parser.add_argument('--verbose', default='False', choices=['True', 'False'],
                    nargs='?', type=str, help='to print progress or not')

args = parser.parse_args()
verbose = True if args.verbose == 'True' else False
benchmark = args.benchmark
benchmark_type = args.benchmark_type
output_path = args.output_path
title = "{}_{}".format(benchmark, benchmark_type) if args.title is None else args.title
name = title if args.name is None else args.name


if benchmark == '101':
    assert benchmark_type in ['nas_cifar10a', 'nas_cifar10b', 'nas_cifar10c',
                                   'protein_structure', 'slice_localization',
                                   'naval_propulsion', 'parkinsons_telemonitoring']

    sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
    sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))
    from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
        FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
    from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C
    data_dir = os.path.join(os.getcwd(), "../nas_benchmarks-development/"
                                         "tabular_benchmarks/fcnet_tabular_benchmarks/")

    if benchmark_type == "nas_cifar10a": # NAS-Bench-101
        max_budget = 108
        b = NASCifar10A(data_dir=data_dir, multi_fidelity=True)
    elif benchmark_type == "nas_cifar10b": # NAS-Bench-101
        max_budget = 108
        b = NASCifar10B(data_dir=data_dir, multi_fidelity=True)
    elif benchmark_type == "nas_cifar10c": # NAS-Bench-101
        max_budget = 108
        b = NASCifar10C(data_dir=data_dir, multi_fidelity=True)
    elif benchmark_type == "protein_structure": # NAS-HPO-Bench
        max_budget = 100
        b = FCNetProteinStructureBenchmark(data_dir=data_dir)
    elif benchmark_type == "slice_localization": # NAS-HPO-Bench
        max_budget = 100
        b = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
    elif benchmark_type == "naval_propulsion": # NAS-HPO-Bench
        max_budget = 100
        b = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
    else:  # benchmark_type == "parkinsons_telemonitoring": # NAS-HPO-Bench
        max_budget = 100
        b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)

    def f(config, budget=None):
        if budget is not None:
            fitness, cost = b.objective_function(config, budget=int(budget))
        else:
            fitness, cost = b.objective_function(config)
        return fitness, cost

    cs = b.get_configuration_space()
    dimensions = len(cs.get_hyperparameters())

elif benchmark == '1shot1':
    assert benchmark_type in ['1', '2', '3']

    sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
    sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))
    from nasbench import api
    from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
    from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
    from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

    nasbench = api.NASBench(os.path.join(os.getcwd(), "../nasbench-1shot1/nasbench_analysis/"
                                                      "nasbench_data/108_e/"
                                                      "nasbench_only108.tfrecord"))
    search_space = eval('SearchSpace{}()'.format(benchmark_type))

    def f(config, budget=None):
        if budget is not None:
            fitness, cost = search_space.objective_function(nasbench, config, budget=int(budget))
        else:
            fitness, cost = search_space.objective_function(nasbench, config)
        fitness = 1 - fitness
        return fitness, cost

    cs = search_space.get_configuration_space()
    dimensions = len(cs.get_hyperparameters())
    max_budget = 108

else:  # benchmark == '201'
    assert benchmark_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']

    sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
    sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))
    from nas_201_api import NASBench201API as API
    from models import CellStructure, get_search_spaces
    data_dir = os.path.join(os.getcwd(), "../nas201/NAS-Bench-201-v1_0-e61699.pth")
    api = API(data_dir)
    search_space = get_search_spaces('cell', 'nas-bench-201')
    config2structure = config2structure_func(4)
    max_budget = 199
    dataset = benchmark_type

    def f(config, budget=max_budget):
        global dataset, api
        structure = config2structure(config)
        arch_index = api.query_index_by_arch(structure)
        if budget is not None:
            budget = int(budget)
        # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
        ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        xoinfo = api.get_more_info(arch_index, 'cifar10-valid', None, True)
        xocost = api.get_cost_info(arch_index, 'cifar10-valid', False)
        info = api.get_more_info(arch_index, dataset, budget, False, True)
        cost = api.get_cost_info(arch_index, dataset, False)

        nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
                'cifar10-valid-train': 25000, 'cifar10-valid-valid': 25000,
                'cifar100-train': 50000, 'cifar100-valid': 5000}
        estimated_train_cost = (xoinfo['train-per-time'] / nums['cifar10-valid-train']) * \
                               (nums['{:}-train'.format(dataset)] / xocost['latency']) * \
                               (cost['latency'] * budget)
        estimated_valid_cost = (xoinfo['valid-per-time'] / nums['cifar10-valid-valid']) * \
                               (nums['{:}-valid'.format(dataset)] / xocost['latency']) * \
                               (cost['latency'] * 1)
        try:
            fitness, cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
        except:
            fitness, cost = info['est-valid-accuracy'], estimated_train_cost + estimated_valid_cost
        fitness = 1 - fitness / 100
        return fitness, cost

    cs = get_configuration_space(4, search_space)
    dimensions = len(cs.get_hyperparameters())


de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=20, mutation_factor=0.5,
        crossover_prob=0.5, strategy='rand1_bin', budget=max_budget)
traj, runtime, history = de.run(generations=args.gens, verbose=verbose)

X = []
X_cs = []
for i in range(len(history)):
    if verbose:
        print("{:<4}/{:<4}".format(i+1, len(history)), end='\r')
    config = de.vector_to_configspace(history[i][0]).get_array()
    X_cs.append(de.vector_to_configspace(history[i][0]).get_array())
    X.append(history[i][0])

X = np.array(X)
X_cs = np.array(X_cs)
X = get_mds(X)
X_cs = get_mds(X_cs)

rgba_colors = generate_colors(X.shape[0], colors=0)
plt.clf()
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(X[:,0], X[:,1], s=15, color=rgba_colors)
ax2.scatter(X_cs[:,0], X_cs[:,1], s=15, color=rgba_colors)
ax1.set_title('DE space [0, 1]')
ax2.set_title('{} parameter space'.format(title))
ax1.set(xlabel="$MDS-X$", ylabel="$MDS-Y$")
ax2.set(xlabel="$MDS-X$", ylabel="$MDS-Y$")
plt.tight_layout()
plt.savefig(os.path.join(output_path, '{}_mds.png'.format(name)), dpi=300)
