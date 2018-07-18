#!/usr/bin/python

# -------------------------------------------------------------------------------
# Script to produce a phylogenetic tree of the tumor population described by
# MyModel.py.
# The phylogenetic tree is drawn using the ete3 library.
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
from ete3 import Tree, TreeStyle, NodeStyle, faces



def build_tree(population, det_lim=1, log=False):
    '''Builds an ete3 Tree object based on the clone phylogeny in the population
        A detection limit can be set which will filter out clones that fall below this limit. The limit is
            one by default, so that only alive clones are taken into account.
        A log-scale can be set which will be used to calculate the node sizes as the log10 of the clone size'''

    def tree_layout(node):
        '''Tree layout function to define the layout of each node within the tree'''
        hex_color = '#%02X%02X%02X' %(node.rgb_color)
        node.img_style["fgcolor"] = hex_color  # set color of node
        node.img_style["size"] = node.weight   # set size of node


    start_clone = population.start_clone
    t = Tree(name=start_clone.ID, dist=0)   # set start clone as root of tree
    if log == True:
        size = 10*np.log10(start_clone.get_family_size())
    else:
        size = start_clone.get_family_size()
    t.add_features(weight=size, rgb_color=start_clone.rgb_color)


    def subtree(clone):
        '''Helper function to generate the subtree for each subclone
            Recursively called to include all subclones situated under given clone'''
        # calculate branch distance as difference between clone and parent birthdays
        distance = clone.birthday - clone.parent.birthday
        s = Tree(name=clone.ID, dist=distance)          # set clone as root of subtree
        if log == True:
            size = 10*np.log10(clone.get_family_size())
        else:
            size = clone.get_family_size()
        s.add_features(weight=size, rgb_color=clone.rgb_color)

        # create copy of subclones list and filter (this avoids the original subclones list to be filtered)
        sub_filtered = clone.subclones[:]
        if det_lim > 0:
            sub_filtered = list(filter(lambda subclone: subclone.get_family_size() >= det_lim, sub_filtered))

        for sub in sub_filtered:
            st = subtree(sub)  # call subtree function recursively for each subclone
            s.add_child(st)
        return s


    # create copy of subclones list and filter (this avoids the original subclones list to be filtered)
    filtered = start_clone.subclones[:]
    if det_lim > 0:
        filtered = list(filter(lambda clone: clone.get_family_size() >= det_lim, filtered))

    for subclone in filtered:
        s = subtree(subclone)
        t.add_child(s)

    # Define TreeStyle
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.rotation = 90  # rotate the tree to get a horizontal one
    ts.layout_fn = tree_layout

    return t, ts
