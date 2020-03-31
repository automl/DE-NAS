'''Differential Evolution class definition
'''

import numpy as np
import ConfigSpace


class DEBase():
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=None,
                 mutation_factor=None, crossover_prob=None, strategy=None, budget=None, **kwargs):
        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # DE related variables
        self.pop_size = pop_size
        self.max_age = max_age
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.budget = budget

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Global trackers
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.age = None
        self.history = []

    def reset(self):
        '''Can be called to reuse the same object for a new DE run
        '''
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.age = None
        self.history = []

    def init_population(self, pop_size=10):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def sample_population(self, size=3):
        selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
        return self.population[selection]

    def boundary_check(self, vector, fix_type='random'):
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

    def vector_to_configspace(self, vector):
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and each component of the
        input vector is in the domain [0, 1].
        '''
        new_config = self.cs.sample_configuration()
        for i, hyper in enumerate(self.cs.get_hyperparameters()):
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

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve_generation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DE(DEBase):
    '''DE object

    Creates and initialises DE with the specified hyperparameters.
    Contains functions to run DE and perform DE operations.

    Parameters
    ----------
    cs : ConfigSpace
        ConfigSpace object defining the parameter space
    f : function
        The objective function that takes a configuration and (optional) budget as input
    dimensions : int
        The dimensionality of the parameter space
    pop_size : int
        Population size of DE for each generation
    max_age : float, optional
        Number of generations after which an individual will be replaced with random individual
    mutation_factor : float
        Scaling factor for mutation (F)
    crossover_prob : float
        Probability for crossover (Cr)
    strategy : str
        Strategies for mutation and crossover concatenated using '_'
        Default of 'rand1_bin' implies
            mutation strategy is 'rand1' and crossover strategy is 'bin' (binomial)
    budget : int, optional
        Can be used to define the budget to be used for evaluation of f()
    '''
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None

    def f_objective(self, x, budget=None):
        '''Objective function that will evaluate x on budget
        '''
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        config = self.vector_to_configspace(x)
        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            fitness, cost = self.f(config, budget=budget)
        else:
            fitness, cost = self.f(config)
        return fitness, cost

    def init_eval_pop(self, budget=None):
        '''Creates new population of 'pop_size' and evaluates individuals
        '''
        self.population = self.init_population(self.pop_size)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []
        for i in range(self.pop_size):
            config = self.population[i]
            self.fitness[i], cost = self.f_objective(config, budget)
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((config.tolist(), float(self.fitness[i]), float(budget or 0)))

        return traj, runtime, history

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        '''Performs the 'current-to-best' type of DE mutation
        '''
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation based on the strategy
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = np.random.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = np.random.randint(0, self.dimensions)
        L = 0
        while ((np.random.rand() < self.crossover_prob) and L < self.dimensions):
            idx = (n+L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover based on the strategy
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        else:
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def selection(self, trials, budget=None):
        '''Carries out a parent-offspring competition given a set of trial population
        '''
        assert len(self.population) == len(trials)
        traj = []
        runtime = []
        history = []
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            fitness, cost = self.f_objective(trials[i], budget)
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = self.population[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(budget or 0)))
        return traj, runtime, history

    def ranked_selection(self, trials, pop_size, budget=None, debug=False):
        '''Returns the fittest individuals from two sets of population
        '''
        # assert len(self.population) == len(trials)
        traj = []
        runtime = []
        history = []
        track = []
        trial_fitness = []
        # pop_size = len(trials)
        for i in range(len(trials)):
            fitness, cost = self.f_objective(trials[i], budget)
            trial_fitness.append(fitness)
            if fitness < self.inc_score:
                self.inc_score = fitness
                self.inc_config = trials[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(budget or 0)))
        if debug:
            print("Ranking {} from {} vs. {}".format(pop_size, self.pop_size, len(trials)))
        tot_pop = np.vstack((self.population, trials))
        tot_fitness = np.hstack((self.fitness, trial_fitness))
        tot_age = np.hstack((self.age, [self.max_age] * len(trials)))
        rank = np.sort(np.argsort(tot_fitness)[:pop_size])
        self.population = tot_pop[rank]
        self.fitness = tot_fitness[rank]
        self.age = tot_age[rank]
        self.pop_size = pop_size
        return traj, runtime, history

    def kill_aged_pop(self, budget=None, debug=False):
        '''Replaces individuals with age older than max_age
        '''
        traj = []
        runtime = []
        history = []
        idxs = np.where(self.age <= 0)[0]
        if len(idxs) == 0:
            return traj, runtime, history
        if debug:
            print("Killing {} individual(s) for budget {}: {}".format(len(idxs), budget, self.age[idxs]))
        new_pop = self.init_population(pop_size=len(idxs))
        for i, index in enumerate(idxs):
            self.population[index] = new_pop[i]
            self.fitness[index], cost = self.f_objective(self.population[index], budget)
            self.age[index] = self.max_age
            if self.fitness[index] < self.inc_score:
                self.inc_score = self.fitness[index]
                self.inc_config = self.population[index]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((self.population[index].tolist(), float(self.fitness[index]),
                            float(budget or 0)))
        return traj, runtime, history

    def evolve_generation(self, budget=None, best=None):
        '''Performs a complete DE evolution, mutation -> crossover -> selection
        '''
        trials = []
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=best)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trials.append(trial)
        trials = np.array(trials)
        traj, runtime, history = self.selection(trials, budget)
        return traj, runtime, history

    def run(self, generations=1, verbose=False, budget=None):
        '''Main function that performs DE optimisation for the specified 'generations'

        Parameters
        ----------
        generations : int
            Number of generations to evolve, or iterations
        verbose : bool
            To print output during optimisation
        budget : int, optional
            Pass budget to override DE budget initialised with
            Allows reuse of the same DE object by calling reset()

        Returns
        -------
        traj : list
            List of fitness value of evaluated configurations
        runtime : list
            List of cost of evaluated configurations
        history : list (of tuples)
            Each tuple contains 3 further elements
                The vector/array form of the configuration (x)
                The fitness value at that iteration
                The budget at which the fitness was evaluated
        '''
        self.traj = []
        self.runtime = []
        self.history = []

        if budget is None:
            budget = self.budget

        if verbose:
            print("Initializing and evaluating new population...")
        self.traj, self.runtime, self.history = self.init_eval_pop(budget=budget)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
            traj, runtime, history = self.evolve_generation(budget=budget)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return np.array(self.traj), np.array(self.runtime), np.array(self.history)
