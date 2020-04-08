'''Runs DE on NAS-Bench-101 and NAS-HPO-Bench
'''

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import json
import pickle
import argparse
import numpy as np
import ConfigSpace

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from scipy.optimize import differential_evolution as DE


def f(config):
    global cs
    config = boundary_check(config)
    config = vector_to_configspace(cs, config)
    fitness, _ = b.objective_function(config)
    return fitness


def vector_to_configspace(cs, vector):
    '''Converts numpy array to ConfigSpace object

    Works when cs is a ConfigSpace object and the input vector is in the domain [0, 1].
    '''
    new_config = cs.sample_configuration()
    for i, hyper in enumerate(cs.get_hyperparameters()):
        if type(hyper) == ConfigSpace.OrdinalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
            param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
        elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
            param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
        else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
            # rescaling continuous values
            param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
            if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                param_value = np.round(param_value).astype(int)   # converting to discrete (int)
        new_config[hyper.name] = param_value
    return new_config


    def boundary_check(vector, fix_type='random'):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        Parameters
        ----------
        vector : array
            The vector describing the individual from the population
        fix_type : str, {'random', 'clip'}
            if 'random', the values are replaced with a random sampling from [0,1)
            if 'clip', the values are clipped to the closest limit from {0, 1}

        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector


def generate_bounds(dimensions):
    bounds = [[(0, 1)] * dimensions][0]
    return bounds


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b", "nas_cifar10c"]
parser.add_argument('--benchmark', default="protein_structure", type=str,
                    help="specify the benchmark to run on from among {}".format(choices))
parser.add_argument('--gens', default=100, type=int, nargs='?',
                    help='(iterations) number of generations for DE to evolve')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../nas_benchmarks-development/"
                                          "tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
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
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='scipy_default', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.benchmark == "nas_cifar10a": # NAS-Bench-101
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10b": # NAS-Bench-101
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10c": # NAS-Bench-101
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "protein_structure": # NAS-HPO-Bench
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization": # NAS-HPO-Bench
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion": # NAS-HPO-Bench
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring": # NAS-HPO-Bench
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

# Initializing DE bounds
bounds = generate_bounds(dimensions)

if args.runs is None:  # for a single run
    if not args.fix_seed:
        np.random.seed(0)
    # Running DE iterations
    _ = DE(f, bounds, popsize=args.pop_size, mutation=args.mutation_factor,
             recombination=args.crossover_prob, init='random', updating='deferred',
             strategy='rand1bin', polish=False, disp=args.verbose)
    # res = DE(f, bounds)
    if 'cifar' in args.benchmark:
        res = b.get_results(ignore_invalid_configs=True)
    else:
        res = b.get_results()
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
        _ = DE(f, bounds, popsize=args.pop_size, mutation=args.mutation_factor,
                 recombination=args.crossover_prob, init='random', updating='deferred',
                 strategy='rand1bin', polish=False, disp=args.verbose)
        # res = DE(f, bounds)
        if 'cifar' in args.benchmark:
            res = b.get_results(ignore_invalid_configs=True)
        else:
            res = b.get_results()
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        if args.verbose:
            print("Run saved. Resetting...")
        # essential step to not accumulate consecutive runs
        b.reset_tracker()
