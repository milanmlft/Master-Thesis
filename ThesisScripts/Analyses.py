#!/usr/bin/python

# -------------------------------------------------------------------------------
# Module that defines the functions used for the analyses of a simulated tumor
# population, defined in MyModel.py
# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def final_data(pop):
    '''Generates a dataframe containing the final properties of every clone in population pop.
        Each row represents a separate clone, each column is a different property'''

    data_keys = ["Clone ID", "Birthday", "q", "Final size", "Family size",
                 "Allele frequency", "Mutations", "Children", "Weight", "RGB color"]
    clone_id = []
    birthday = []
    mutrate = []
    final_size = []
    family_size = []
    af = []
    num_acc_mutations = []
    num_children = []
    weight = []
    rgb_color = []

    for clone in pop.clones:
        clone_id.append(clone.ID)
        birthday.append(clone.birthday)
        mutrate.append(clone.mutrate)
        final_size.append(clone.size)
        fam_size = clone.get_family_size()
        family_size.append(fam_size)
        af.append(fam_size / pop.size)
        num_acc_mutations.append(clone.num_mutations)
        num_children.append(len(clone.subclones))
        weight.append(clone.weight)
        rgb_color.append(clone.rgb_color)

    data_values = [clone_id, birthday, mutrate, final_size, family_size, af,
                   num_acc_mutations, num_children, weight, rgb_color]
    zipped = list(zip(data_keys, data_values))
    data = dict(zipped)
    df = pd.DataFrame(data)
    df.set_index('Clone ID', inplace=True)  # setting Clone IDs as index
    df = df[['Birthday', 'q', 'Final size', 'Family size', 'Allele frequency', 'Mutations',
             'Children', 'Weight', 'RGB color']]   # rearranging order of columns

    return df



def sample(data, size, det_lim):
    '''
    Performs multinomial sampling on a given dataset. Calculates the new sampled clone sizes, corresponding family sizes and new Allele Frequencies.

    Parameters
    ----------
    * data : pandas.DataFrame, dataset in final_data() format containing simulated tumor clones
    * size : int, total number of cells to be sampled (should be smaller than original population size)
    * det_lim : int, detection limit, minimal family size of clones to be included in sample

    Returns
    -------
    * sampled_clones : pandas.DataFrame, new DataFrame containing only sampled clones, same format as final_data() but with additional columns: "sampled_size", "sampled_fam_size" and "sampled_AF"

    '''
    N_sample = size  # total size of sampled cells

    # filter clones on detection limit
    sampled_clones = data.loc[data["Family size"] >= det_lim]
    N_total = np.sum(sampled_clones["Final size"].values)

    size_sum = 0     # sum of sizes of all preceding clones, starting at 0
    sampled_sum = 0  # sum of sampled sizes of all preceding clones, starting at 0

    # calculate sampled clone sizes
    sampled_clones = sampled_clones.assign(sampled_size = np.zeros(len(sampled_clones), dtype=int))  # add new empty column

    for index, clone in sampled_clones.iterrows():
        n = clone["Final size"]
        x = np.random.binomial(N_sample - sampled_sum, n/(N_total - size_sum))
        sampled_clones.loc[index, "sampled_size"] = int(x)

        size_sum += n
        sampled_sum += x

    # calculate sampled family sizes
    sampled_clones = sampled_clones.assign(sampled_fam_size = np.zeros(len(sampled_clones), dtype=int))  # add new empty column

    for index, clone in sampled_clones.iterrows():
        mask = sampled_clones.index.str.contains(str(index)+'.', regex=False)  # mask to find clone's children
        sampled_fam_size = clone["sampled_size"] + np.sum(sampled_clones.loc[mask, "sampled_size"].values)
        sampled_clones.loc[index, "sampled_fam_size"] = sampled_fam_size

    # filter out clones with sampled fam size == 0
    sampled_clones = sampled_clones.loc[sampled_clones["sampled_fam_size"] > 0]

    # calculate sampled Allele Frequencies
    sampled_clones = sampled_clones.assign(sampled_AF = lambda clone: clone.sampled_fam_size / N_sample)

    return sampled_clones


#-----------------------------------------------------------------------------------------------------------
# Single population analyses
# The following functions are used to analyze a single simulated population
#-----------------------------------------------------------------------------------------------------------


def num_mutations_evolution(population):
    '''Return DataFrame containing the evolution of clone.num_mutations over the generations and a figure
        containing the corresponding chart'''
    dic = {}   # temporary dictionary to store data

    for clone in population.clones:
        n = clone.num_mutations
        if n == 1:
            key = "%i mutation" %(n)
        else:
            key = "%i mutations" %(n)

        if key not in dic.keys():
            dic[key] = sum(x.size_vec for x in population.clones if x.num_mutations == n)

    df = pd.DataFrame(dic)
    df = df.rename_axis("generation")

    # creating the evolution chart
    fig = plt.figure()
    ax = df.plot(colormap='Accent', style='d--', xticks=df.index)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cell count')
    ax.set_yscale('log')

    return df, fig



def num_mutations_distribution(data, sampled=False, det_lim=1):
    '''Returns a dataframe containing the number of mutations and the total number of cells carrying that
        number of mutations
        Uses a DataFrame in the format given by final_data() as input'''

    data = data.loc[data['Final size'] >= det_lim].iloc[1:]  # filter out clones with size < det_lim
    max_n = data['Mutations'].max()  # maximal number of mutations in one clone

    dic = {}  # initialize empty dictionary to store data
    dic["# mutations"] = []
    dic["Cell count"] = []

    for n in range(1, max_n+1):
        if sampled:
            size = np.sum(data[data['Mutations'] == n]['sampled_size'])
        else:
            size = np.sum(data[data['Mutations'] == n]['Final size'])
        dic["# mutations"].append(n)
        dic["Cell count"].append(size)

    df = pd.DataFrame(dic)
    df.set_index("# mutations", inplace=True)
    return df



def avg_num_mutations(num_mut_distr):
    '''Returns the average number of mutations per clone, weighted by the clone size.
        Takes a DataFrame of the format given by num_mutations_distribution as input.'''
    nmd = num_mut_distr
    w_sizes = []
    for i in range(nmd.index.min(), nmd.index.max()+1):
        weighted_size = i * nmd.loc[i]
        w_sizes.append(weighted_size)

    w_sizes = np.array(w_sizes)
    avg = np.sum(w_sizes) / np.sum(nmd.values)

    return avg



def fit_cumulative_mutations(data, sampled=False, det_lim=1, plot=False):
    '''
    Calculates the cumulative number of mutations per allelic frequency (M(f)) and plots it in function of 1/f
        and then fits the data using linear regression to test whether M(f) is linear with 1/f

    Extracts allele frequencies from a final_data() type of dataframe
    Filters out clones with family size < detection limit (det_lim, default 1)

    If sampled == True, uses sampled data

    If plot == True, returns a figure containing the plot (default: False)

    Returns
    -------
    * raw_data: raw data consisting of a DataFrame with 1/f and corresponding Cumulative count
    * r_value: R2 value of the linear regression between the cumulative count and 1/f
    * fig (if plot==True): plot of M(f) and the linear regression fit
    '''
    # filter out start clone and clones with size < det_lim
    data = data.loc[data['Family size'] >= det_lim].iloc[1:]
    if sampled:
        f = data["sampled_AF"].values
    else:
        f = data["Allele frequency"].values   # don't take start clone into account
    f_inv = 1/f
    f_inv_sorted = np.sort(f_inv)
    u, counts = np.unique(f_inv_sorted, return_counts=True)
    cf = np.cumsum(counts)

    # calculate the linear regression fit between cf and u
    slope, intercept, r_value, p_value, std_err = stats.linregress(u, cf)
    r = round(r_value, 3)

    if plot:
        #coordinates to display R2 text on graph
        rx = 0.8*max(u)
        ry = 1.2*(intercept + slope*rx)
        fig = plt.figure()
        ax = plt.plot(u, cf, marker='o', markersize=5, lw=0, color='black')
        plt.plot(u, intercept + slope*u, 'r')
        plt.xlabel("Inverse allelic frequency")
        plt.ylabel("Cumulative number of mutations")
        new_ticks = np.linspace(min(u), max(u), num=5)
        xticklabels = ['1/%f' %(i) for i in 1/new_ticks]
        plt.xticks(new_ticks, xticklabels)
        plt.text(rx, ry, r'$R^2=$ %s'%(r))

    return r_value


def heterogeneity(data, sampled=False, det_lim=1):
    '''Calculates the heterogeneity of a population, based on Simpson's diversity index.
        Option to set a detection limit (det_lim, default 1)'''

    data = data.loc[data["Final size"] >= det_lim].iloc[1:]
    if sampled:
        sizes = data["sampled_size"].values
    else:
        sizes = data["Final size"].values # individual detected clone sizes
    N = np.sum(sizes)  # total size of detected clones
    h = 1 - np.sum((sizes/N)**2)
    return h



def reconstruct_mutational_timeline(data, alpha, sampled=False, det_lim=1):
    '''
    Reconstructs mutational history of the population from allele frequencies,
        in units of population size doublings.

    Parameters:
    -----------
    * data : final_data() dataframe format
    * alpha : death fraction of the analyzed population
    * sampled : use sampled data? (yes if True)
    * det_lim : detection limit used to filter out small clones

    Returns 3 arrays:
    -------
    * reconstructed timepoints at which mutations occurred
    * reconstructed population sizes
    * errors: deviation of the reconstructed timepoint from the real one, absolute value
    '''

    clones = data.loc[data["Final size"] >= det_lim].iloc[1:]

    rec_birthdays = []
    rec_popsizes = []
    errors = []

    for index, clone in clones.iterrows():
        if sampled:
            af = clone["sampled_AF"]
        else:
            af = clone["Allele frequency"]

        # reconstructing population size from allele frequency of clone
        rec_N = 1/af

        # calculating birthday of clone from reconstructed population size (according to N(T) = 2**T)
        # in units of number of population size doublings
        rec_T = np.log(rec_N)/np.log(2)
        rec_popsizes.append(rec_N)
        rec_birthdays.append(rec_T)

        # calculating the error on the reconstructed birthday
        # real birthday has to be converted to dimensionless time (T = t/tD = t*y/ln(2))
        real_T = clone["Birthday"]
        real_T = real_T*np.log(2-alpha)/np.log(2)
        error = np.abs(real_T - rec_T)
        errors.append(error)

    rec_popsizes = np.array(rec_popsizes)
    rec_birthdays = np.array(rec_birthdays)
    errors = np.array(errors)

    return rec_birthdays, rec_popsizes, errors


#-----------------------------------------------------------------------------------------------------------
# Multiple simulations analyses
# The following functions are used to analyze an ensemble of multiple populations simulated with
# identical parameters.
#-----------------------------------------------------------------------------------------------------------


def get_max_AFs(populations_data, sampled=False, det_lim=1):
    '''Returns an array containing the maximum subclone allele frequencies from a given set of populations
        Uses the final_data() function from ThesisScripts.Analyses to get the allele frequencies
        Option to set a detection limit (det_lim, default 1)'''
    max_afs = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        data = populations_data[i]
        mutations = data.loc[data['Family size'] >= det_lim].iloc[1:]
        if sampled:
            max_af = mutations['sampled_AF'].max()
        else:
            max_af = mutations['Allele frequency'].max()
        max_afs[i] = max_af

    return max_afs


def get_avg_AFs(populations_data, sampled=False, det_lim=1):
    '''Returns an array containing the average subclone allele frequencies from a given set of populations
        Option to set a detection limit (det_lim, default 1)'''
    avg_afs = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        data = populations_data[i]
        mutations = data.loc[data['Family size'] >= det_lim].iloc[1:]
        if sampled:
            avg_af = mutations['sampled_AF'].mean()
        else:
            avg_af = mutations['Allele frequency'].mean()
        avg_afs[i] = avg_af

    return avg_afs



def get_fit_r_values(populations_data, sampled=False, det_lim=1):
    '''Returns an array containing the R2 values of fitting the cumulative number of mutations against
        1/f (with f = allele frequency), calculated by fit_cumulative_mutations()'''

    r_values = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        data = populations_data[i]
        if sampled:
            r = fit_cumulative_mutations(data, sampled=True, det_lim=det_lim, plot=False)
        else:
            r = fit_cumulative_mutations(data, sampled=False, det_lim=det_lim, plot=False)
        r_values[i] = r

    return r_values



def get_heterogeneity(populations_data, sampled=False, det_lim=1):
    '''Returns an array containing the Simpson's diversity indices of the given populatons, calculated by
        the heterogeneity() function
        Option to set a detection limit (det_lim=1)'''

    h_arr = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        if sampled:
            h = heterogeneity(populations_data[i], sampled=True, det_lim=det_lim)
        else:
            h = heterogeneity(populations_data[i], sampled=False, det_lim=det_lim)
        h_arr[i] = h

    return h_arr



def get_mutation_distributions(populations_data, sampled=False, det_lim=1):
    '''Returns a DataFrame containing the cell counts for each number of mutations from the set of populations.
        Uses the num_mutations_distribution() function from ThesisScripts.Analyses to get the distribution of
            mutations.
        Optional to set a detection limit (det_lim, default 1)'''

    df = pd.DataFrame()  # initialize empty dataframe to store the cell counts for each population

    for i in range(len(populations_data)):
        data = populations_data[i]
        if sampled:
            nmd = num_mutations_distribution(data, sampled=True, det_lim=det_lim)
        else:
            nmd = num_mutations_distribution(data, sampled=False, det_lim=det_lim)
        nmd.columns = ["Population %s" %(i)]
        df = df.join(nmd, how='outer')  # join dataframes together as union
    return df.T


def get_total_mutations(populations_data, det_lim=1):
    '''Returns an array containing the total number of detectable mutations within
        each population'''
    m_arr = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        data = populations_data[i]
        data = data.loc[data["Family size"] > det_lim]
        m = len(data) - 1  # don't count ancestral clone
        m_arr[i] = m

    return m_arr



def get_reconstruction_errors(populations_data, alpha, sampled=False, det_lim=1):
    '''Returns an array containing the median errors of reconstructing the mutational timeline for the given
        set of populations, i.e. the MAD of the reconstructed timepoints around their real value.
        Option to use sampled data
        Option to set a detection limit (det_lim, default 1)'''

    med_errors = np.empty(len(populations_data))

    for i in range(len(populations_data)):
        if sampled:
            errors = reconstruct_mutational_timeline(populations_data[i], alpha=alpha, sampled=True, det_lim=det_lim)[2]
        else:
            errors = reconstruct_mutational_timeline(populations_data[i], alpha=alpha, sampled=False, det_lim=det_lim)[2]
        med_errors[i] = np.median(errors)

    return med_errors
