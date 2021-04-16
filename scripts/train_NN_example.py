#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_NN.py: script to train a dense feedforward NN with one hidden layer.
Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""

import sys_path
from source.NN import train_NN_1hl


# set sample_size
num_samples = 40

# determine number of hidden nodes
num_hidden_nodes = 200

# set path to training data
sim_filepath = "/home/klara/ml/NN/data/simulations/franz/"
sim_filename = "sim_n_40_rep_100000_rho_35_theta_random-100"

# set path to save NN
save_filepath = "../data/saved_NN/"

# train dense feedforward NN with one hidden layer:
train_NN_1hl(
    num_samples, num_hidden_nodes, sim_filepath, sim_filename, save_filepath
)
