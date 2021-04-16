#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt_coefficients.py: Functions to compute and save coefficients for the linear
estimator of the mutation rate, minimising the variance or the MSE. Functions to
compute optimal coefficients numerically are added.
Created on Tue Dec  1 16:20:36 2020

@author: Franz Baumdicker, Klara Burger
"""
import numpy
from sympy import symbols
import sys_path
from source.base.calculate import (
    optimal_coeff_V,
    optimal_coeff_MSE,
    r,
    r2,
    sigma_matrix,
)


# function to compute and save coefficients of MMSEE (Minimal MSE Estimator)
def coeff_v(num_sample, filepath):
    # Saving the optimal coefficients of the MVUE estimator into a .npy file.
    for n in range(num_sample, num_sample + 1):
        filename = filepath + "opt_coeff_mvue_" + str(n)
        # optimal coefficients for a given number of samples:
        opt_coeff = optimal_coeff_V(n)
        x = symbols("x")
        # save the results as npy file
        numpy.save(filename, opt_coeff(x))


# function to compute and save coefficients of MMSEE (Minimal MSE Estimator)
def coeff_mse(num_sample, filepath):
    # Saving the optimal coefficients of the MVUE estimator into a .npy file.
    for n in range(num_sample, num_sample + 1):
        filename = filepath + "opt_coeff_mmsee_" + str(n)
        # optimal coefficients for a given number of samples:
        opt_coeff = optimal_coeff_MSE(n)
        x = symbols("x")
        # save the results as npy file
        numpy.save(filename, opt_coeff(x))


# coefficients of MVUE by Fu
def coeff_Fu(n, theta):
    r_2 = r2(n)
    r_1 = r(n)
    sig = sigma_matrix(n)
    coeff = numpy.zeros((n - 1,))
    A = numpy.diag(r_2) + theta * sig
    Ainv = numpy.linalg.inv(A)
    coeff = (r_2 @ Ainv) / (r_1 @ Ainv @ numpy.transpose(r_1))
    return coeff


# coefficients of MMSEE by Futschik
def coeff_Futschik(n, theta):
    r_2 = r2(n)
    r_1 = r(n)
    sig = sigma_matrix(n)
    coeff = numpy.zeros((n - 1,))
    if theta == 0:
        coeff = numpy.zeros((n - 1,))
    else:
        A = numpy.diag(r_2) / theta + sig + numpy.transpose(r_1) @ r_1
        coeff = numpy.linalg.solve(A, r_2)
    return coeff
