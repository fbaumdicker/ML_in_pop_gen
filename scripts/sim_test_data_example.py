#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_test_data.py: script to simulate data for testing various
estimators of the mutation rate (model-based and model-free).
Created on Tue Dec  1 15:35:13 2020

@author: Klara Burger, Franz Baumdicker
"""
import sys_path
from source.base.simulate import simulate_data


# determine characteristics for simulation
num_samples = 40  # sample size
rep = 1000  # number of replicates
rho = 0  # recombination rate
my_filepath = "../data/simulations/rho_0/"  # state filepath to save data sets in

# simulate datasets for theta=1,...,40:
for i in range(1, 41):
    # simulate data:
    theta = float(i)
    simulate_data(
        num_samples,
        rep,
        theta,
        rho,
        save_genotype_matrix=False,
        filepath=my_filepath,
        save_ts=False,
    )
    print("Data set for theta =", theta, "simulated.")
