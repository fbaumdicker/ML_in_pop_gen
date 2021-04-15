#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_coeff_mvue.py: compute & save coeff. for MVUE for fixed sample size.
Created on Tue Dec  1 16:20:36 2020

@author: Franz Baumdicker, Klara Burger
"""
import sys_path
from source.base.opt_coefficients import coeff_v

# determine sample size
n = 3

# determine filepath to save coefficients
filepath = "../data/precalculated_optimal_coefficients/"

# compute and save coefficients for MVUE
coeff_v(n, filepath)
