import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

num_cores = 8


class pool(object):
    """represents the population."""

    def __init__(self, dim, poolsize, low_lim, high_lim, fitness=None):
        super(pool, self).__init__()
        self.dim = dim
        self.poolsize = poolsize
        self.poolarray = np.array([])
        self.fitnessmap = np.array([])
        self.history = []
        self.low_lim = low_lim
        self.high_lim = high_lim
        if fitness:
            self.fitness = fitness

    def createpool(self):
        poolarray = []
        for i in range(self.poolsize):
            poolarray.append(chromosome(self.dim, self.low_lim, self.high_lim))
        self.poolarray = poolarray

    # operations
    def mutate(self, geneit):
        randind = np.random.choice(self.dim)
        geneit.chromosome_[randind] = np.random.rand() + np.random.randint(self.low_lim, self.high_lim)#TODO might fail to satisfy constraints
        return geneit

    def crossover(self, chromosome_a, chromosome_b):
        randind = np.random.choice(self.dim)
        parta = chromosome_a.chromosome_[:randind]
        partb = chromosome_b.chromosome_[:randind]

        chromosome_a.chromosome_[:randind] = partb
        chromosome_b.chromosome_[:randind] = parta
        return chromosome_a, chromosome_b

    def create_chromosome(self):
        return chromosome(self.dim, self.low_lim, self.high_lim)

    def fitness(self, chromosome):

        result = np.abs(1000 - (chromosome.chromosome_[0] + 0 * chromosome.chromosome_[1] ** 3)) + np.sum(chromosome.chromosome_) * 0.
        return result

    def calc_fitness(self,use_parallel=False):
        if use_parallel:
            errors = Parallel(n_jobs=num_cores)(delayed(self.fitness)(i) for i in tqdm(self.poolarray))
        else:
            errors = []
            for i in self.poolarray:
                errors.append(self.fitness(i))
        self.fitnessmap = errors

    def passnextgen(self):

        self.calc_fitness()

        #hyper params
        numelit = 1

        prob_motation = 0.6
        prob_crossover = 0.4
        prob_new = 0.2

        nextpool = []
        elit_indx = np.argpartition(self.fitnessmap, numelit)[:numelit]

        for i in elit_indx: #pass the elites w/o touching them
            nextpool.append(deepcopy(self.poolarray[i]))

        prob = np.random.rand()

        for ind, cgene in enumerate(self.poolarray):
            # cgene = deepcopy(cgene)

            if ind in elit_indx:
                continue

            else:
                if prob < prob_motation:
                    cgene = self.mutate(cgene)
                if prob < prob_crossover:
                    othergene = np.random.choice(self.poolarray)
                    cgene, othergene = self.crossover(cgene, othergene)
                if prob < prob_new:
                    cgene = self.create_chromosome()

            nextpool.append(cgene)

        self.poolarray = nextpool[0:self.poolsize]

    def iterate(self, numiter):
        history = []
        for i in range(numiter):

            self.passnextgen() #TODO no roulete well selection, currently uses uniform selection of chromosomes and operations to pass it on next gen
            history.append(min(self.fitnessmap))
            if (min(self.fitnessmap)) < 0.00001: #tolerance #TODO make tolerance parametric
                self.calc_fitness() #fitness is reached
                print('already found the soln!')

                bestindx = np.squeeze(np.argpartition(self.fitnessmap, 1)[:1])
                bestgene = self.poolarray[bestindx].geneit
                self.history = history
                return bestgene, self.fitness(self.poolarray[bestindx])

        bestindx = np.squeeze(np.argpartition(self.fitnessmap, 1)[:1]) #elite genes
        bestgene = self.poolarray[bestindx].chromosome_
        self.history = history
        return bestgene, self.fitness(self.poolarray[bestindx])

    def show_hist(self):
        """displays the change of fitness across generations"""
        from matplotlib import pyplot as plt
        plt.plot(self.history)
        plt.title('fitness')
        plt.xlabel('generations')
        plt.ylabel('fitness')
        plt.show()


class chromosome(object):
    """individual representation of a solution candidate, ie chromosome"""

    def __init__(self, dim, low_lim, high_lim):
        super(chromosome, self).__init__()
        self.dim = dim
        self.high_lim = high_lim
        self.low_lim = low_lim
        self.chromosome_ = self.create_chromosome()

    def create_chromosome(self):
        return np.random.rand(self.dim) + np.random.randint(self.low_lim, self.high_lim, self.dim) #TODO: each gene shold have its own ...
                                                                                                        # limits / constraints
