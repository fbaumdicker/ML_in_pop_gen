#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_adaptive_NN.py: script to train an adaptive dense feedforward NN with
one hidden layer.

Adaptive: devide the range of values of theta the NN is trained for in classes
and according to the performance of the NN per class in comparison to all model
based estimators and the linear NN put more weight on classes of comparatively
poor performance. Extent of weight increase depends on deviation from the best
estimator the NN is compared to.

Created on Tue Mar 9 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""

import sys_path
from source.adaptive_NN import train_adaptive_NN_1hl

# set parameters for model definition
num_samples = 40  # set sample_size
num_hidden_nodes = 200  # determine number of hidden nodes
num_NN = 1  # number of NN to be trained

# set parameters for adaptive training procedure
num_class = 6  # number of classes of loss function, weights per class are
# adapted during training
tol = 0.04  # tolerance for defining classes, typically tol = 0.04
max_it = 200  # number of max. updates of weights for classes of loss
sloppiness = 0.02  # determines percentage NN is allowed to perform worse
# than best comparing estimator

# set path to training data
sim_filepath = "../data/simulations/"
sim_filename = "sim_n_40_rep_200000_rho_0_theta_random-100"
theta_min = 0  # minimal theta in training data
theta_max = 100  # maximal theta in training data

# set path to saved linear NN (adaptive NN is to be compared to linear NN)
linear_NN_filepath = "../data/saved_NN/"
linear_NN_filename = "Linear_NN_sim_n_40_rep_200000_rho_0_theta_random-100"

# set path to save NN
save_filepath = "../data/saved_NN/"

# train adaptive dense feedforward NN with one hidden layer:
train_adaptive_NN_1hl(
    num_samples,
    num_hidden_nodes,
    num_class,
    num_NN,
    theta_min,
    theta_max,
    tol,
    max_it,
    sloppiness,
    sim_filepath,
    sim_filename,
    linear_NN_filepath,
    linear_NN_filename,
    save_filepath,
)
