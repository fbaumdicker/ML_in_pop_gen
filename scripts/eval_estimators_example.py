#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_estimators.py: script to evaluate Watterson, ItV, ItMSE, MVUE, MMSEE,
NN & linear NN on a given series of data sets. It can be set, if results should
be saved or not. For increasing the computational speed, the evaluation of
model based estimators can be turned of.
Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""
import sys_path
from source.evaluate_estimators import evaluate_estimators

# first state some characteristics about the series of data sets:
num_sample = 40  # state sample size of the datasets
rep_test_data = 1000  # state number of repetitions in each test dataset
rep_training_data = 200000  # state number of repetitions in training data set for NN
rho = 0  # state recombination rate in each dataset
filepath = "../data/simulations/"  # state the filepath to the data sets
num_hidden_nodes = 200  # number of hidden nodes of the NN

# evaluate Watterson, ItV, ItMSE, NN & linear NN the stated series of data sets
# and save the results:
evaluate_estimators(
    num_sample,
    rep_test_data,
    rho,
    num_hidden_nodes,
    rep_training_data,
    filepath,
    eval_model_based_est=True,
    save=True,
)
