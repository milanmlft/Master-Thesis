#!/usr/bin/python

# -------------------------------------------------------------------------------
# Module that defines the Population and Clone classes which can be used to
# simulate a tumor population.
# Described in more detail in Model.ipynb (in Notebooks)
# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
from ThesisScripts.Analyses import final_data


class Population:
    '''
    The Population class will be used to simulate the whole tumor population, which consists of clones and their subclones

    Attributes:
    -----------
        * gen: the generation or age of the population, starting at 0
        * size_lim: size to which the population grows
        * mutfactor: factor with which mutation rates of subclones will increase to simulate mutator phenotype (default 1)
        * death_frac: the fraction of the total population that will die
        * size: total population size, which is the sum of all clones and subclones (calculated with get_size() method)
        * size_vec: vector to store evolution of population sizes (start size = self.size = 1)
        * mut_vec: vector to store number of mutations that occured in each generation (initial value at generation 0 = 0)
        * mutprob = starting mutation probability; the probability of generating a mutant for the start clone
        * start_clone: the initial clone within the population, represented by the Clone class, starting with 1 clone
        * clones: a list to store all clones present in the population, starting with the inital clone
        * selection: if > 0: determines the variance in selective advantage for the clones
    '''
    def __init__(self, size_lim, mutprob, mutfactor=1, death_frac=0, selection=0):
        self.gen = 0
        self.size_lim = size_lim
        self.mutfactor = mutfactor
        self.death_frac = death_frac
        self.size = 1
        self.size_vec = [self.size]
        self.mut_vec = [0]
        self.start_clone = Clone(self, "A", q=mutprob, parent=None, num_mutations=0, weight=1)  # first clone carries 0 mutations and has no advantage
        self.clones = [self.start_clone]
        self.selection=selection

    def get_size(self):
        '''Calculates the total population size as the sum of all individual (sub)clone sizes'''
        N = sum((x.size for x in self.clones))
        return N

    def divisions_deaths(self):
        '''
        Calculates the number of dividing cells for each clone at the current generation, based on a multinomial distribution of the total population
        Clone weights are used as a measure of selective advantage when sampling dividers
        The number of deaths is calculated stochastically in a similar way but using a multinomial hypergeometric distribution.
        '''
        N = self.get_size()       # total population size
        alpha = self.death_frac   # fraction of deaths
        D = int(round(alpha*N))   # total number of deaths, ROUNDED to get an integer number of deaths

        size_sum = 0   # sum of sizes of all preceding clones, starting at 0
        div_sum = 0    # sum of dividers of all preceding clones, starting at 0
        death_sum = 0  # sum of deaths of all preceding clones, starting at 0

        # for selection: only calculate N_weighted when there is selection involved
        if self.selection == 0:
            N_weighted = N
        else:
            N_weighted = sum((x.weight*x.size for x in self.clones))

        size_sum_w = 0 # weighted sum of preceding clone sizes, starting at 0

        # do stochastic sampling for all clones except last one (fixed)
        for clone in self.clones[:-1]:
            n = clone.size
            w = clone.weight
            nw = n*w  # weighted size

            # dividers
            if div_sum >= N:    # total number of dividers cannot exeed total population size!
                clone.dividers = 0
            else:
                try:
                    clone.dividers = np.random.binomial(N - div_sum, nw/(N_weighted - size_sum_w))
                except:
                    clone.dividers = np.random.binomial(N - div_sum, 1)

            # deaths
            if death_sum >= D:  # total number of deaths cannot exceed D
                clone.deaths = 0
            else:
                clone.deaths = np.random.hypergeometric(n, N - size_sum - n, D - death_sum)

            size_sum += n              # update size_sum with current clone size
            size_sum_w += nw
            div_sum += clone.dividers  # update div_sum with current clone dividers
            death_sum += clone.deaths  # update death_sum with current clone deaths

        # for last clone: number of dividers and deaths is fixed
        last_clone = self.clones[-1]
        last_clone.dividers = N - div_sum
        last_clone.deaths = D - death_sum

    def simulate(self):
        '''
        Simulates the growth of the tumor population
        After the simulation has completed, all clones that are present are stored in the clones list
        This list will be used to analyze the clone sizes and heterogeneity
        '''
        while self.size < self.size_lim:

            self.gen += 1

            new_clones = []    # reset new_clones list at each generation
            mutations = 0

            self.divisions_deaths()

            # grow and mutate clones
            for clone in self.clones:
                clone.evolve()
                new_clones += clone.new_subs  # add new clones
                mutations += clone.mutators

            self.mut_vec.append(mutations)

            # update clones list
            self.clones += new_clones

            self.size = self.get_size()
            self.size_vec.append(self.size)


        # convert vector lists to numpy arrays
        self.mut_vec = np.array(self.mut_vec)
        self.size_vec = np.array(self.size_vec)

        # adjust length of clone size vectors so they all have equal lengths
        # by inserting 0's at the start of the vector (size 0 before they occur)
        for clone in self.clones:
            while len(clone.size_vec) < len(self.size_vec):
                clone.size_vec.insert(0,0)
            clone.size_vec = np.array(clone.size_vec)         # convert to numpy array




class Clone:
    '''
    The Clone class represents a distinct subpopulation of genetic identical cells within the total tumour population.
    Each mutation within a clone leads to a new subclone, which is in turn a clone that grows and can generate subclones of itself.

    Attributes:
        * population: points to the Population class objects to which the clone belongs
        * parent: the parent clone
        * birthday: generation at which the clone was generated
        * size: amount of cells within the clone, start size is always 1
        * dividers: number of cells that will divide (calculated by divisions_deaths method in Population class)
        * deaths: number of cells that will die (calculated by divisions_deaths method in Population class)
        * mutators: number of cells that will mutate
        * subclones: list of subclones that originated from mutations within the current clone, each subclone is in turn a new clone
        * ID: each clone is identified by an ID, this ID is composed of the ancestor ID and a number
        * mutrate: clone specific mutation rate, passed on and increased (in case of mutator phenotype) from its ancestor
        * num_mutations: number of mutations carried by this clone, this number remains constant
        * subid: a number that keeps track of how many subclones have occured within the clone, and is used to assign a unique ID to new subclones
        * size_vec: a vector to store the evolution of the sizes of the clone,
        * weight: the weight that reflects the selective advantage of the clone
        * rgb_color: fixed color for the clone to be used in visualizations
    '''
    def __init__(self, population, ID, q, parent, num_mutations, weight):
        self.population = population
        self.parent = parent
        self.birthday = self.population.gen
        self.size = 1
        self.dividers = 0
        self.deaths = 0
        self.mutators = 0
        self.subclones = []
        self.ID = ID
        self.mutrate = q
        self.num_mutations = num_mutations
        self.subid = 0
        self.size_vec = [self.size]
        self.weight = weight
        self.rgb_color = self.set_color()

    def evolve(self):
        '''
        Lets the clone grow
        The divisions  and deaths are STOCHASTIC: the number of dividing and dyingcells (self.dividers) is
            calculated for each clone by the divisions_deaths() method in the Population class
        The clone size grows with this number of dividing cells and decreases with the number of deaths
        Calculates the number of cells that will mutate (mutators) within the clone, the number of which
            is chosen randomly according to a binomial distribution from the clone dividers
        Each mutation leads to a new clone and decreases the clone size
        For each new subclone, a new mutation rate q is calculated as an increase to that of its parent (in case of mutator phenotype)
        For each new subclone, a new weight is sampled from a gamma distribution around the weight of the ancestor (in case of selection)
        The new size is appended to the size vector
        Returns the list new_subs, containing the new subclones generated at the current generation, which can
            then be used by the Population class to keep track of all new subclones
        '''
        q = self.mutrate                       # clone specific mutation probability
        mutfactor = self.population.mutfactor  # factor with which q will increase for subclones
        new_q = mutfactor*q                    # calculate new mutation rate for subclones, based on parent's mutation rate
        if new_q > 1:
            # report when subclones have reached the maximally attainable mutation probability of 1
            print("Subclones of clone %s have reached q = 1" %(self.ID))
            new_q = 1

        self.new_subs = []   # list of new subclones generated by the current mutation step

        self.mutators = np.random.binomial(self.dividers, q)

        new_num_mutations = self.num_mutations + 1    # number of mutations for new subclones

        # parameters for sampling of new weight from gamma distribution
        if self.population.selection > 0:
            scale = self.population.selection
            shape = self.weight / scale   # so that mean = shape * scale == self.weight (parent clone weight)


        for i in range(self.mutators):
            # calculate new weight for every new subclone, based on parent's weight, from gamma distribution
            if self.population.selection > 0:
                new_weight = np.random.gamma(shape, scale)
            else:
                new_weight = self.weight   # in case of no selection

            new_ID = self.ID + "." + str(self.subid + i)  # assign new ID as extension of parent's ID

            self.new_subs.append(Clone(self.population, ID=new_ID, q=new_q, parent=self,
                                       num_mutations=new_num_mutations, weight=new_weight))

        self.subclones += self.new_subs  # add new subclones to the total subclones list
        self.subid += self.mutators

        self.size = self.size + self.dividers - self.deaths - self.mutators   # update clone size
        self.size_vec.append(self.size)                                       # update clone size vector

        return self.new_subs


    def get_family_size(self):
        '''Returns the total size of the clone and all its descendants.
            Function is recursively called to access all descendants.'''

        total = self.size

        for sub in self.subclones:
            total += sub.get_family_size()

        return total


    def get_family_size_vec(self):
        '''Returns a vector that contains the family size of the clone per generation
            This vector is the sum of the clone size vector and the size vectors of all its descendants'''

        family_size_vec = np.array(self.size_vec)

        for sub in self.subclones:
            family_size_vec = family_size_vec + sub.get_family_size_vec()

        return family_size_vec


    def set_color(self):
        '''Generates a color in RGB format for the clone, based on a lighter tint of its parent color
            For the start_clone (parent == None), the color is set to black (0, 0, 0)'''

        if self.parent == None:
            rgb = (0, 0, 0)

        elif self.parent == self.population.start_clone:
            # for the mutations in the start clone: generate new random color
            c = lambda: np.random.randint(0, 256)  # generate random value between 0 and 255
            rgb = (c(), c(), c())

        else:
            factor = 0.1  # factor with which to change the color tint
            old_rgb = self.parent.rgb_color
            new_rgb = lambda x: tuple(int(factor*(255 - c) + c) for c in x)  # function to generate lighter RGB tint
            rgb = new_rgb(old_rgb)
        return rgb




def run_simulations(path_prefix, n, size_lim, mutprob, mutfactor, death_frac, selection):
    '''
    Runs multiple Population simulations with the same parameters.

    Parameters:
    -----------
    * path_prefx : prefix of the filepath to target folder to save data
    * n : int, number of simulations to run
    * size_lim : int, size limit of populations
    * mutprob : float, starting mutation rate for populations
    * mutfactor : float, factor with which to raise the mutation rate for each subclone
    * death_frac : float, death rate for populations
    * selection : float, selection factor for populations

    Returns:
    --------
    * data_list : list, contains the final_data (see Analyses.py) dataframe of each simulated population
    '''

    data_list = []

    for i in tqdm(range(n)):
        pop = Population(size_lim, mutprob, mutfactor, death_frac, selection)
        pop.simulate()
        data = final_data(pop)
        # pickle data
        file_path = path_prefix + 'population_' + str(i)  + '.pkl.gz'
        data.to_pickle(file_path, compression='gzip')
        data_list.append(data)

    return data_list
