'''Runs DE on NAS-Bench-201
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))

import json
import pickle
import argparse
import numpy as np
import ConfigSpace

from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces

from denas import DE


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
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


def calculate_regrets(history, runtime):
    assert len(runtime) == len(history)
    global dataset, api, de, max_budget

    regret_test = []
    regret_validation = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1
    for i in range(len(history)):
        config, valid_regret, budget = history[i]
        valid_regret = valid_regret - y_star_valid
        if valid_regret <= inc:
            inc = valid_regret
            config = de.vector_to_configspace(config)
            structure = config2structure(config)
            arch_index = api.query_index_by_arch(structure)
            info = api.get_more_info(arch_index, dataset, max_budget, False, False)
            test_regret = (1 - (info['test-accuracy'] / 100)) - y_star_test
        regret_validation.append(inc)
        regret_test.append(test_regret)
    res = {}
    res['regret_test'] = regret_test
    res['regret_validation'] = regret_validation
    res['runtime'] = np.cumsum(runtime).tolist()
    return res


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def find_nas201_best(api, dataset):
    arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
    _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
    return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                    help='choose the dataset')
parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                    help='maximum number of nodes in the cell')
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nas201/NAS-Bench-201-v1_0-e61699.pth",
                    help='specifies the path to the benchmark data')
parser.add_argument('--pop_size', default=20, type=int, nargs='?', help='population size')
strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                    'currenttobest1_bin', 'randtobest1_bin',
                    'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                    'currenttobest1_exp', 'randtobest1_exp']
parser.add_argument('--strategy', default="rand1_bin", choices=strategy_choices,
                    type=str, nargs='?',
                    help="specify the DE strategy from among {}".format(strategy_choices))
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                    help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                    help='probability of crossover')
parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                    help='maximum wallclock time to run DE for')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='de', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
max_budget = args.max_budget
dataset = args.dataset

# Directory where files will be written
output_path = os.path.join(args.output_path, args.dataset, args.folder)
os.makedirs(output_path, exist_ok=True)

# Loading NAS-201
api = API(args.data_dir)
search_space = get_search_spaces('cell', 'nas-bench-201')

# Parameter space to be used by DE
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)

y_star_valid, y_star_test = find_nas201_best(api, dataset)
inc_config = cs.get_default_configuration().get_array().tolist()

# Custom objective function for DE to interface NASBench-201
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
            'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
            'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
    estimated_train_cost = (xoinfo['train-per-time'] / nums['cifar10-valid-train']) * \
                           (nums['{:}-train'.format(dataset)] / xocost['latency']) * \
                           (cost['latency'] * budget)
    estimated_valid_cost = (xoinfo['valid-per-time'] / nums['cifar10-valid-valid']) * \
                           (nums['{:}-valid'.format(dataset)] / xocost['latency']) *\
                           (cost['latency'] * 1)
    try:
        fitness, cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
    except:
        fitness, cost = info['est-valid-accuracy'], estimated_train_cost + estimated_valid_cost
    fitness = 1 - fitness / 100
    return fitness, cost


# Initializing DE object
de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, budget=args.max_budget)


if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
    res = calculate_regrets(history, runtime)
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()
else:  # for multiple runs
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        # Running DE iterations
        traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
        res = calculate_regrets(history, runtime)
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        de.reset()

save_configspace(cs, output_path)
