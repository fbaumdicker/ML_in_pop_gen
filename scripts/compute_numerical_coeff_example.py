#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_coeff_mvue.py: compute & save coeff. for MVUE for fixed sample size.
Created on Tue Dec  1 16:20:36 2020

@author: Franz Baumdicker, Klara Burger
"""
import sys_path
from source.base.opt_coefficients import coeff_Fu, coeff_Futschik
import source.base.calculate
import sys

if len(sys.argv) != 3:
    print("You can set n and theta by typing:\n" +
        "python3 compute_numerical_coeff_example.py n theta")
    # set sample size
    n = 10
    # set true mutation rate theta
    theta = 40
else:
    # set sample size
    n = int(sys.argv[1])
    # set true mutation rate theta
    theta = float(sys.argv[2])
    
print("Using sample size n = ", n, " and true mutation rate theta = ", theta, "\n")


# compute and print coefficients for MVUE
print("optimal parameters as used for Fu's estimator:\n")
print(coeff_Fu(n, theta))

# compute and print coefficients for MMSEE
print("\noptimal parameters as used for Futschik's estimator:\n")
print(coeff_Futschik(n, theta))


