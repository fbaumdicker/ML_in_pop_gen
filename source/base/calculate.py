#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate.py: Functions to calculate optimal coefficients (s.t. a linear
estimator minimises the variance or the MSE) to estimate the mutation rate.
Functions to compute optimal coefficients numerically are also added.
Created on Tue Dec  1 16:20:04 2020
@author: Klara Burger, Franz Baumdicker
"""

import numpy
from sympy import diff, solve, symbols


# helper functions for the symbolic calculation of optimal coefficients
# function for calculating the n-th harmonic number
def harmonic(n):
    harmonic_number = 0
    for i in range(1, n):  # calculation of the harmonic number
        harmonic_number = harmonic_number + (1 / i)
    return harmonic_number


# Formulas taken from Fu (1995)
def beta1(n, i):
    return 2 * n * (harmonic(n + 1) - harmonic(i)) / ((n - i + 1) * (n - i)) - (
        2 / (n - i)
    )


def sigma_ij(n, i, j):
    if i + j < n:
        return (beta1(n, i + 1) - beta1(n, i)) / 2
    elif i + j == n:
        return (
            (harmonic(n) - harmonic(i)) / (n - i)
            + (harmonic(n) - harmonic(j)) / (n - j)
            - (beta1(n, i) + beta1(n, j + 1)) / 2
            - 1 / (i * j)
        )
    else:
        return (beta1(n, j) - beta1(n, j + 1)) / 2 - 1 / (i * j)


def sigma_ii(n, i):
    if i < (n / 2):
        return beta1(n, i + 1)
    elif i == (n / 2):
        return 2 * (harmonic(n) - harmonic(i)) / (n - i) - (1 / (i ** 2))
    else:
        return beta1(n, i) - 1 / (i ** 2)


# function to calculate the optimal choice of parameters for a given number of
# samples and variable theta:
def optimal_coeff_V(num_sample, unbiased=True):
    def coeff(theta):
        # create symbolic variables
        C = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            C[i - 1] = symbols("c_%d" % i)
            lagrange = symbols("lagrange")

        sum1 = 0
        sum2 = 0

        # presteps for variance calculation
        for i in range(1, num_sample):
            sum1 = sum1 + (i ** 2) * (C[i - 1] ** 2) * (
                (theta / i) + sigma_ii(num_sample, i) * (theta ** 2)
            )

        for i in range(2, num_sample):
            for j in range(1, i):
                sum2 = sum2 + i * j * C[i - 1] * C[j - 1] * sigma_ij(
                    num_sample, i, j
                ) * (theta ** 2)

        # variance
        vari = sum1 + 2 * sum2

        # minimization of the variance using Lagrangian function
        D = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            D[i - 1] = diff(vari, C[i - 1]) + lagrange

        D.append(sum(C) - 1)
        Eqn = list([] for i in range(0, num_sample))
        for i in range(1, num_sample + 1):
            Eqn[i - 1] = D[i - 1]  # ==0

        # print("Equations to solve:")
        # print(Eqn)

        # solution of the minimization problem
        C.append(lagrange)
        # Extracting optimal values of the coefficients c_i into the list C_i
        C_sol = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            C_sol[i - 1] = [sol[C[i - 1]] for sol in solve(Eqn, C, dict=True)]
            C_sol[i - 1] = C_sol[i - 1][0]
            # print(C_sol[i-1][0])

        # resulting coefficients a_i (a_i = i* c_i):
        A_sol = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            A_sol[i - 1] = i * C_sol[i - 1]
        return A_sol

    return coeff


# function to calculate the optimal choice of parameters to further reduce the
# variance via a bias-variance tradeoff, i.e. minimizing the MSE. The function
# depends on a given number of samples and variable theta.
def optimal_coeff_MSE(num_sample, unbiased=False):
    def coeff(theta):
        # create symbolic variables
        A = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            A[i - 1] = symbols("a_%d" % i)

        sum1 = 0
        sum2 = 0
        sum3 = 0

        # presteps for variance calculation
        for i in range(1, num_sample):
            sum1 = sum1 + (A[i - 1] ** 2) * (
                (theta / i) + sigma_ii(num_sample, i) * (theta ** 2)
            )

        for i in range(2, num_sample):
            for j in range(1, i):
                sum2 = sum2 + A[i - 1] * A[j - 1] * sigma_ij(num_sample, i, j) * (
                    theta ** 2
                )

        for i in range(1, num_sample):
            sum3 = sum3 + (theta / i) * A[i - 1]

        # MSE (mean squared error)
        vari = sum1 + 2 * sum2 + +((sum3 - theta) ** 2)

        # minimization of the MSE
        D = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            D[i - 1] = diff(vari, A[i - 1])

        Eqn = list([] for i in range(0, num_sample - 1))
        for i in range(1, num_sample):
            Eqn[i - 1] = D[i - 1]  # ==0

        # print("Equations to solve:")
        # print(Eqn)

        # solution of the minimization problem

        # Extracting optimal values of the coefficients a_i into the list A_i
        A_sol = list([] for i in range(1, num_sample))
        for i in range(1, num_sample):
            A_sol[i - 1] = [sol[A[i - 1]] for sol in solve(Eqn, A, dict=True)]
            A_sol[i - 1] = A_sol[i - 1][0]
            # print(C_sol[i-1][0])

        return A_sol

    return coeff


# helper functions for the numerical calculation of optimal coefficients
# function to compute a vector with the i-th harmonic number in entry i
def harmonic_vec(n):
    harmonic_vector = numpy.zeros((n + 1,))
    for i in range(1, n + 1):
        harmonic_vector[i] = harmonic(i + 1)
    return harmonic_vector


# function to compute an 1x(n-1)-matrix with 1/(i+1) in entry i
def r(n):
    r = numpy.zeros((1, n - 1))
    for i in range(0, n - 1):
        r[0, i] = 1 / (i + 1)
    return r


# function to compute a vector with n-1 entries, 1/(i+1) in entry i
def r2(n):
    r = numpy.zeros(
        n - 1,
    )
    for i in range(0, n - 1):
        r[i] = 1 / (i + 1)
    return r


# Formulas taken from Fu (1995) in matrix formulation
def beta_vec(n):
    beta_vector = numpy.zeros((n - 1,))
    for i in range(0, n - 1):
        beta_vector[i] = (
            2 * n * (harmonic(n + 1) - harmonic(i)) / ((n - i + 1) * (n - i))
        ) - (2 / (n - i))
    return beta_vector


def sigma_matrix(n):
    sigma_ma = numpy.zeros((n - 1, n - 1))
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                if i < (n / 2):
                    sigma_ma[i - 1, i - 1] = beta1(n, i + 1)
                elif i == (n / 2):
                    sigma_ma[i - 1, i - 1] = 2 * (harmonic(n) - harmonic(i)) / (
                        n - i
                    ) - (1 / (i ** 2))
                else:
                    sigma_ma[i - 1, i - 1] = beta1(n, i) - 1 / (i ** 2)
            elif i + j < n and i > j:
                sigma_ma[i - 1, j - 1] = (beta1(n, i + 1) - beta1(n, i)) / 2
            elif i + j < n and j > i:
                sigma_ma[i - 1, j - 1] = (beta1(n, j + 1) - beta1(n, j)) / 2
            elif i + j == n and i > j:
                sigma_ma[i - 1, j - 1] = (
                    (harmonic(n) - harmonic(i)) / (n - i)
                    + (harmonic(n) - harmonic(j)) / (n - j)
                    - (beta1(n, i) + beta1(n, j + 1)) / 2
                    - 1 / (i * j)
                )
            elif i + j == n and j > i:
                sigma_ma[i - 1, j - 1] = (
                    (harmonic(n) - harmonic(j)) / (n - j)
                    + (harmonic(n) - harmonic(i)) / (n - i)
                    - (beta1(n, j) + beta1(n, i + 1)) / 2
                    - 1 / (i * j)
                )
            elif i + j > n and i > j:
                sigma_ma[i - 1, j - 1] = (beta1(n, j) - beta1(n, j + 1)) / 2 - 1 / (
                    i * j
                )
            else:
                sigma_ma[i - 1, j - 1] = (beta1(n, i) - beta1(n, i + 1)) / 2 - 1 / (
                    i * j
                )
    return sigma_ma
