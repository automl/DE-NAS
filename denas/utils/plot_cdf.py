'''Script to plot empirical CDF of final test regret
    for multiple runs on the benchmarks'''

import os
import json
import sys
import pickle
import argparse
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
seaborn.set_style("ticks")

from matplotlib import rcParams
rcParams["font.size"] = "30"
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['figure.edgecolor'] = 'k'
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = 'k'
rcParams['axes.grid.which'] = 'both'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1

rcParams['ytick.major.size'] = 12
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 6
rcParams['ytick.minor.width'] = 1
rcParams['xtick.major.size'] = 12
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 6
rcParams['xtick.minor.width'] = 1

marker=['x', '^', 'D', 'o', 's', 'h', '*', 'v', '<', ">"]
linestyles = ['-', '--', '-.', ':']


def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = collections.OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    # creates a dataframe where the rows are indexed based on time
    # fills with NA for missing values for the respective timesteps
    merged = pd.DataFrame(frame_dict)
    # ffill() acts like a fillna() wherein a forward fill happens
    # only remaining NAs for in the beginning until a value is recorded
    merged = merged.ffill()

    performance = merged.get_values()  # converts to a 2D numpy array
    time_ = merged.index.values        # retrieves the timestamps

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


parser = argparse.ArgumentParser()

parser.add_argument('--bench', default='101', type=str, nargs='?',
                    choices=['101', '1shot1', '201'], help='select benchmark')
parser.add_argument('--ssp', default=None, type=int, nargs='?')
parser.add_argument('--path', default='./', type=str, nargs='?',
                    help='path to encodings or jsons for each algorithm')
parser.add_argument('--n_runs', default=10, type=int, nargs='?',
                    help='number of runs to plot data for')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the plot will be saved')
parser.add_argument('--type', default="wallclock", type=str, choices=["wallclock", "fevals"],
                    help='to plot for wallclock times or # function evaluations')
parser.add_argument('--name', default="comparison", type=str,
                    help='file name for the PNG plot to be saved')
parser.add_argument('--title', default="benchmark", type=str,
                    help='title name for the plot')
parser.add_argument('--limit', default=1e7, type=float, help='wallclock limit')
parser.add_argument('--regret', default='validation', type=str, choices=['validation', 'test'],
                    help='type of regret')

args = parser.parse_args()
path = args.path
n_runs = args.n_runs
plot_type = args.type
plot_name = args.name
regret_type = args.regret
benchmark = args.bench
ssp = args.ssp

if benchmark == '1shot1' and ssp is None:
    print("Specify \'--ssp\' from {1, 2, 3} for choosing the search space for NASBench-1shot1.")
    sys.exit()

if benchmark == '101':
    methods = [
               # ("bohb", "BOHB"),
               # ("hyperband", "HB"),
               # ("tpe", "TPE"),
               ("regularized_evolution", "RE"),
               # ("de_pop10", "DE $pop=10$"),
               ("de_pop20", "DE")]
               # ("de_pop50", "DE $pop=50$"),
               # ("de_pop100", "DE $pop=100$")]
elif benchmark == '201':
    methods = [
               # ("tpe", "TPE"),
               ("regularized_evolution", "RE"),
               ("de_pop20", "DE")]

else:
    methods = [
               # ("BOHB", "BOHB"),
               # ("HB", "HB"),
               # ("TPE", "TPE"),
               ("RE", "RE"),
               # ("DE_pop10", "DE $pop=10$"),
               ("DE_pop20", "DE")]
               # ("DE_pop50", "DE $pop=50$"),
               # ("DE_pop100", "DE $pop=100$")]

# plot limits
min_time = np.inf
max_time = 0
min_regret = 1
max_regret = 0

# plot setup
colors = ["C%d" % i for i in range(len(methods))]
plt.clf()

# looping and plotting for all methods
for index, (m, label) in enumerate(methods):
    regret = []
    runtimes = []
    for k, i in enumerate(np.arange(n_runs)):
        try:
            if benchmark in ['101', '201']:
                res = json.load(open(os.path.join(path, m, "run_%d.json" % i)))
            else:
                res = json.load(open(os.path.join(path, m, str(ssp), "run_%d.json" % i)))
            no_runs_found = False
        except Exception as e:
            print(m, i, e)
            no_runs_found = True
            continue
        regret_key =  "regret_validation" if regret_type == 'validation' else "regret_test"
        runtime_key = "runtime"
        _, idx = np.unique(res[regret_key], return_index=True)
        idx.sort()
        regret.append(np.array(res[regret_key])[idx])
        runtimes.append(np.array(res[runtime_key])[idx])

    if not no_runs_found:
        # finds the latest time where the first measurement was made across runs
        t = np.max([runtimes[i][0] for i in range(len(runtimes))])
        min_time = min(min_time, t)
        te, time = fill_trajectory(regret, runtimes, replace_nan=1)

        idx = time.tolist().index(t)
        te = te[idx:, :]
        time = time[idx:]

        # Clips off all measurements after 10^7s
        idx = np.where(time < args.limit)[0]

        print("{}. Plotting for {}".format(index, m))
        print(len(regret), len(runtimes))

        ## final regret
        te = te[idx][-1]

        n, bins, patches = plt.hist(te, density=True, histtype='step', cumulative=True,
                                    linewidth=4, label=label, color=colors[index],
                                    linestyle=linestyles[index % len(linestyles)])
        patches[0].set_xy(patches[0].get_xy()[:-1])

        # Stats to dynamically impose limits on the axes of the plots
        min_regret = min(min_regret, np.min(te))
        max_regret = max(max_regret, np.max(te))

plt.xscale("log")
plt.tick_params(which='both', direction="in")
plt.legend(loc='lower right', framealpha=1, prop={'size': 35, 'weight': 'bold'})
plt.title(args.title)
plt.ylabel("probability", fontsize=50)
plt.xlabel("final {} regret".format(regret_type), fontsize=50)
plt.grid(which='both', alpha=0.5, linewidth=0.5)
print(os.path.join(args.output_path, '{}.png'.format(plot_name)))
plt.savefig(os.path.join(args.output_path, '{}.png'.format(plot_name)),
            bbox_inches='tight', dpi=300)
