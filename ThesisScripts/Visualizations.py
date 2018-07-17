#!/usr/bin/python

# -------------------------------------------------------------------------------
# Module that defines the functions used for the data visualization of a simulated
# tumor population, defined in MyModel.py
#
# LAST Changes (20/04/2018, from Visualizations.ipynb):
#
#   * Added notebook functions
#
# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def size_plots(population, det_lim=0):
    '''Plots evolution of total population size and all individual clone sizes'''

    fig = plt.figure(figsize=(14, 14))

    # First subplot: total population size and start-clone individual size
    ax1 = plt.subplot(211)
    ax1.plot(population.size_vec, marker='*', label="Total Population")
    start_hex_rgb = '#%02X%02X%02X' %(population.start_clone.rgb_color)  # get start-clone color
    start_lab = "Clone %s" %population.start_clone.ID                    # create label for start clone
    ax1.plot(population.start_clone.size_vec, marker='.', color=start_hex_rgb, label=start_lab)
    ax1.set_ylabel("Cell count")
    ax1.set_yscale('log')
    ax1.set_title('Total population and start clone')
    ax1.legend()
    ax1.set_xticks(range(0, population.gen+1, 1))

    # Second sublot: plot subclones sizes
    clones = population.clones[1:]   # excluding start-clone
    if det_lim > 0:
        clones = list(filter(lambda subclone: subclone.get_family_size() >= det_lim, clones))

    ax2 = plt.subplot(212)

    for clone in clones:
        hex_rgb_col = '#%02X%02X%02X' %(clone.rgb_color)  # convert RGB color to hex format to be used in plot
        lab = "Clone %s" %clone.ID
        ax2.plot(clone.size_vec, marker='.', color=hex_rgb_col, label=lab)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Cell count")
    ax2.set_yscale('log')
    ax2.set_title('Subclones')
    ax2.set_xticks(range(0, population.gen+1, 1))

    return fig



def family_size_plots(population, det_lim=0):
    '''Creates plots of the clone family sizes per generation'''

    clones = population.clones[1:]   # don't take start clone into account (has same plot as total population)

    if det_lim > 0:
        clones = list(filter(lambda subclone: subclone.get_family_size() >= det_lim, clones))

    fig = plt.figure()

    for clone in clones:
        family_size_vec = clone.get_family_size_vec()
        hex_rgb_col = '#%02X%02X%02X' %(clone.rgb_color)  # convert clone RGB color to hex format
        lab = "Clone %s" %clone.ID                        # create label from clone ID
        plt.plot(family_size_vec, marker='.', color=hex_rgb_col, label=lab)

    plt.xlabel("Generation")
    plt.ylabel("Family size")
    plt.yscale('log')
    plt.xticks(range(0, population.gen+1, 1))

    return fig



def mutations_barplot(population, log=False):
    '''makes barplot of number of mutations in each generation'''
    plt.figure()
    plt.bar(range(population.gen+1), population.mut_vec, color='k')
    plt.title('Number of mutations per generation')
    plt.ylabel('# mutations occurred')
    plt.xlabel('Generation')
    if log == True:
        plt.yscale('log')
    plt.xticks(np.arange(0, population.gen+1, 1))
    plt.show()
