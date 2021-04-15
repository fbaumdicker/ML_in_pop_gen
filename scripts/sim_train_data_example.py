#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_train_data.py: script to simulate data for training neural networks.
Created on Tue Dec  1 15:35:13 2020

@author: Klara Burger, Franz Baumdicker
"""

import sys_path
from source.base.simulate import simulate_data

# determine characteristics for simulation. Set the rho=-1 to simulate data
# with a random recombination rate rho in (0,50)
num_samples = 40  # sample size
rep = 10000  # number of replicates
rho = 35  # recombination rate, rho = -1 indicates that rho should be random
my_filepath = "../data/simulations/"  # filepath to save data

# mutation rate, theta=0 indicates that the data should be simulated by
# using a random theta in each repetition:
theta = 0

# simulate data:
simulate_data(
    num_samples,
    rep,
    theta,
    rho,
    save_genotype_matrix=True,
    filepath=my_filepath,
    save_ts=False,
)
