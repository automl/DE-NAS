'''Runs DE on NAS-Bench-1shot1
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nasbench/'))
sys.path.append(os.path.join(os.getcwd(), '../nasbench-1shot1/'))

import json
import pickle
import argparse
import numpy as np

from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3

from denas import DE


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


parser = argparse.ArgumentParser()
parser.add_argument('--search_space', default=None, type=str, nargs='?',
                    help='specifies the benchmark')
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', type=str, nargs='?',
                    default="../nasbench-1shot1/nasbench_analysis/nasbench_data/"
                            "108_e/nasbench_only108.tfrecord",
                    help='specifies the path to the tabular data')
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
parser.add_argument('--max_budget', default=108, type=str, nargs='?',
                    help='maximum wallclock time to run DE for')
parser.add_argument('--verbose', default='True', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='de', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

nasbench = api.NASBench(args.data_dir)

if args.search_space is None:
    spaces = [1, 2, 3]
else:
    spaces = [int(args.search_space)]

for space in spaces:
    print('##### Search Space {} #####'.format(space))
    search_space = eval('SearchSpace{}()'.format(space))
    y_star_valid, y_star_test, inc_config = (search_space.valid_min_error,
                                             search_space.test_min_error, None)
    # Parameter space to be used by DE
    cs = search_space.get_configuration_space()
    dimensions = len(cs.get_hyperparameters())

    output_path = os.path.join(args.output_path, args.folder)
    os.makedirs(output_path, exist_ok=True)

    # Objective function for DE
    def f(config, budget=None):
        if budget is not None:
            fitness, cost = search_space.objective_function(nasbench, config, budget=int(budget))
        else:
            fitness, cost = search_space.objective_function(nasbench, config)
        fitness = 1 - fitness
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
        fh = open(os.path.join(output_path,
                               'DE_{}_ssp_{}_seed_0.obj'.format(args.run_id, space)), 'wb')
        pickle.dump(search_space.run_history, fh)
        fh.close()
    else:  # for multiple runs
        for run_id, _ in enumerate(range(args.runs), start=args.run_start):
            if not args.fix_seed:
                np.random.seed(run_id)
            if args.verbose:
                print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
            # Running DE iterations
            traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
            fh = open(os.path.join(output_path,
                                   'DE_{}_ssp_{}_seed_{}.obj'.format(run_id, space, run_id)), 'wb')
            pickle.dump(search_space.run_history, fh)
            fh.close()
            if args.verbose:
                print("Run saved. Resetting...")
            # essential step to not accumulate consecutive runs
            de.reset()
            search_space.run_history = []

    save_configspace(cs, output_path)
