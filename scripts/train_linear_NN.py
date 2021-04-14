#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_linear_NN.py: script to train a linear dense feedforward NN.
Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""
import sys_path
from source.NN import train_linear_NN


# set sample_size
num_samples = 40

# set path to training data
sim_filepath = "../data/simulations/"
sim_filename = "sim_n_40_rep_200000_rho_0_theta_random-100"

# set path to save NN
save_filepath = "../data/saved_NN/"

# train dense feedforward NN with one hidden layer:
train_linear_NN(num_samples, sim_filepath, sim_filename, save_filepath)
