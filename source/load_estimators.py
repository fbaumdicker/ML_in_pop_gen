#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_estimators.py: script to load features of evaluated Watterson, ItV,
ItMSE, NN and linear NN.
Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""

import numpy


def load_features(estimator, n, rho, theta_min, theta_max):
    # load estimators:
    loaded = numpy.load(
        "../data/features_estimators/n_"
        + str(n)
        + "_theta_"
        + str(theta_min)
        + "_"
        + str(theta_max)
        + "_rho_"
        + str(rho)
        + "_"
        + estimator
        + ".npy"
    )
    return loaded
