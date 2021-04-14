#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_estimators.py: script to save features of evaluated Watterson, ItV,
ItMSE, NN and linear NN.
Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""

import numpy


def save_features(
    estimator, estimator_name, n, rho, theta_min, theta_max
):
    # save estimators and main characteristics of the used data sets:
    numpy.save(
        "../data/features_estimators/n_"
        + str(n)
        + "_theta_"
        + str(theta_min)
        + "_"
        + str(theta_max)
        + "_rho_"
        + str(rho)
        + "_"
        + estimator_name,
        estimator,
    )
