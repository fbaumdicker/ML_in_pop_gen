#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate.py: Function to simulate genealogical trees via msprime.
In particular, it simulates SFS, genotype matrix, tree lengths...
Created on Mon Dec 14 14:16:36 2020

@author: Franz Baumdicker, Klara Burger
"""
import numpy
import msprime
import collections
import random
import pickle


# function to simulate SFS
def simulate_data(
    num_sample,
    rep,
    theta,
    rho,
    save_genotype_matrix=True,
    filename1="",
    filepath="",
    save_ts=False,
):

    multi_SFS = []  # list to save the SFS
    multi_total_length = []  # list to save the total tree lengths
    multi_G = []  # list to save the genotype matrices
    multi_theta = []  # list to save theta used for simulating
    multi_rho = []  # list to save rho used for simulating
    multi_ts = []  # list to save the tree sequences

    # check if dataset is simulated for training:
    train = False
    theta_str = str(theta)
    if theta == 0:
        train = True
        theta_str = "random-100"

    rho_train = False
    rho_str = str(rho)
    if rho == -1:
        rho_train = True
        rho_str = "random-50"

    # simulate a datasets of size rep
    for i in range(0, rep):
        # if training data, take in each iteration new theta to simulate
        if train:
            theta = random.uniform(0, 100)

        if rho_train:
            rho = random.uniform(0, 50)
            multi_rho.append(rho)

        # simulate the coalescent tree with msprime
        tree_sequence = msprime.simulate(
            sample_size=num_sample,
            Ne=0.25,
            length=1,
            recombination_rate=rho,
            mutation_rate=theta,
            random_seed=None,
        )
        if save_ts:
            multi_ts.append(tree_sequence)

        # get mean total tree length and save as entry of multi_total_length
        mean_tot_branch_length = 0
        for tree in tree_sequence.trees():
            mean_tot_branch_length += tree.total_branch_length * (
                tree.interval[1] - tree.interval[0]
            )
        multi_total_length.append(mean_tot_branch_length)

        # get genotype matrix
        G = tree_sequence.genotype_matrix()
        # potentially save the genotype matrix:
        if save_genotype_matrix:
            multi_G.append(G)
        assert G.shape[1] == num_sample

        # calculate site frequency spectrum from genotype matrix
        # sum over columns of the genotype matrix
        a = G.sum(axis=1)
        # site frequency spectrum
        S = numpy.zeros((num_sample - 1,), dtype=int)
        for i in range(0, num_sample - 1):
            S[i] = collections.Counter(a)[i + 1]

        # save the SFS and the theta used for simulation
        multi_SFS.append(S)
        multi_theta.append(theta)

    # save SFS and mean total tree length in datafile
    if filename1 == "":
        filename1 = (
            filepath
            + "sim_n_"
            + str(num_sample)
            + "_rep_"
            + str(rep)
            + "_rho_"
            + rho_str
            + "_theta_"
            + theta_str
        )
    numpy.savez(
        filename1,
        multi_SFS=multi_SFS,
        multi_G=multi_G,
        multi_total_length=multi_total_length,
        multi_theta=multi_theta,
        multi_rho=multi_rho,
    )

    if save_ts:
        filehandler = open(filename1 + ".ts", "wb")
        pickle.dump(multi_ts, filehandler)
