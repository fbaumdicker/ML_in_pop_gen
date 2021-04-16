#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_based_estimators.py: Functions to calculate Watterson, MVUE, MMSEE (both
numerically and exact), ItV and ItMSE.
Created on Tue Dec  1 16:20:36 2020

@author: Franz Baumdicker, Klara Burger
"""

import numpy
import sys_path
from sympy import lambdify, sympify, N
from sympy.abc import x
from pathlib import Path
from source.base.calculate import harmonic, r, r2, sigma_matrix
from source.base.opt_coefficients import coeff_Fu, coeff_Futschik


# define function to compute watterson estimator
def watterson(S):
    num_sample = S.size + 1
    coefficient_watterson = 1 / harmonic(
        num_sample
    )  # calculating the coefficients for the Watterson estimator
    # wattersons estimation for theta
    watterson_est = sum(S) * coefficient_watterson
    return watterson_est


# function to load the optimal coefficients of MVUE depending on theta=x
def load_symb_optimal_coeffs_mvue(
    samplesize,
    filepath="./data/precalculated_optimal_coefficients/",
    filenameprefix="opt_coeff_mvue_",
):
    file = Path("filepath + filenameprefix + str(samplesize)")
    if file.exists():
        coeff = numpy.load(
            filepath + filenameprefix + str(samplesize) + ".npy",
            allow_pickle=True,
        )
        exact_symb_coeff = sympify(coeff)
        print(exact_symb_coeff)
        symb_coeff = []
        for i in range(0, len(exact_symb_coeff)):
            symb_coeff.append(N(exact_symb_coeff[i]))
        return lambdify(x, sympify(symb_coeff), "mpmath")
    else:
        print(
            "ERROR: Optimal coefficients for MVUE are not precalculated yet \
            for this specific sample size. Do so with the script \
            compute_coeff_mvue.py in source"
        )


# function to load optimal coefficients of MMSEE depending on theta=x
def load_symb_optimal_coeffs_mmsee(
    samplesize,
    filepath="./data/precalculated_optimal_coefficients/",
    filenameprefix="opt_coeff_mmsee_",
):
    file = Path("filepath + filenameprefix + str(samplesize)")
    if file.exists():
        coeff = numpy.load(
            filepath + filenameprefix + str(samplesize) + ".npy",
            allow_pickle=True,
        )
        exact_symb_coeff = sympify(coeff)
        print(exact_symb_coeff)
        symb_coeff = []
        for i in range(0, len(exact_symb_coeff)):
            symb_coeff.append(N(exact_symb_coeff[i]))
        return lambdify(x, sympify(symb_coeff), "mpmath")
    else:
        print(
            "ERROR: Optimal coefficients for MMSEE are not precalculated yet \
            for this specific sample size. Do so with the script \
            compute_coeff_mmsee.py in source"
        )


# MMSEE
def MMSEE(S, theta):
    N, num_sample = S.shape
    num_sample = num_sample + 1
    # print(num_sample)
    coeff = coeff_Futschik(num_sample, theta)
    # print(coeff)
    est = numpy.zeros((N,))
    for j in range(0, N):
        for i in range(0, num_sample - 1):
            est[j] = est[j] + S[j, i] * coeff[i]
    # print(est)
    return est


# MVUE
def MVUE(S, theta):
    N, num_sample = S.shape
    num_sample = num_sample + 1
    coeff = coeff_Fu(num_sample, theta)
    # print(coeff[0])
    # A_sol2 = coeff(theta)
    # A_sol = numpy.array(A_sol2, dtype=float)
    est = numpy.zeros((N,))
    for j in range(0, N):
        for i in range(0, num_sample - 1):
            est[j] = est[j] + S[j, i] * coeff[0][i]
    # print(est)
    return est


# ItMSE
def ItMSE(S, tol):
    tol = tol
    N, n = S.shape
    n = n + 1
    r_1 = r(n)
    r_2 = r2(n)
    sig = sigma_matrix(n)
    theta_hat = numpy.zeros((N,))
    init_theta_hat = numpy.zeros((N,))
    it = numpy.zeros((N,))
    for i in range(0, N):
        coeff = numpy.zeros((n - 1,))
        theta_hat[i] = (
            numpy.sum(
                S[
                    i,
                ]
            )
            / harmonic(n)
        )
        init_theta_hat[i] = theta_hat[i]
        error = 1
        if (
            numpy.sum(
                S[
                    i,
                ]
            )
            == 0
        ):
            theta_hat[i] = 0
        else:
            while error > tol:
                it[i] = it[i] + 1
                theta_hat_old = theta_hat[i]
                A = numpy.diag(r_2) / theta_hat_old + sig + numpy.transpose(r_1) @ r_1
                coeff = numpy.linalg.solve(A, r_2)
                theta_hat[i] = numpy.sum(
                    coeff
                    * numpy.transpose(
                        S[
                            i,
                        ]
                    )
                )
                if theta_hat[i] < 0:
                    theta_hat[i] = 0
                    break
                error = numpy.abs(theta_hat[i] - theta_hat_old)
    return theta_hat


# ItV
def ItV(S, tol):
    tol = tol
    N, n = S.shape
    n = n + 1
    r_2 = r2(n)
    r_1 = r(n)
    sig = sigma_matrix(n)
    theta_hat = numpy.zeros((N,))
    it = numpy.zeros((N,))
    for i in range(0, N):
        coeff = numpy.zeros((n - 1,))
        theta_hat[i] = (
            numpy.sum(
                S[
                    i,
                ]
            )
            / harmonic(n)
        )
        error = 1
        while error > tol:
            it[i] = it[i] + 1
            theta_hat_old = theta_hat[i]
            A = numpy.diag(r_2) + theta_hat_old * sig
            Ainv = numpy.linalg.inv(A)
            coeff = (numpy.transpose(r_1) * Ainv) / (r_1 @ Ainv @ numpy.transpose(r_1))
            theta_hat[i] = numpy.sum(
                coeff
                * numpy.transpose(
                    S[
                        i,
                    ]
                )
            )
            error = numpy.abs(theta_hat[i] - theta_hat_old)
    return theta_hat
