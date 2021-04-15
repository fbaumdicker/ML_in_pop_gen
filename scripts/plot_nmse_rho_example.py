#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_nmse_rho.py: script to plot the normalised MSE vs. the recombination rate
rho=0,1,...,50,1000 for the NN trained with rho=0,35,1000 fixed, a linear NN
and an adaptive NN trained with variable rho and Wattterson and ItMSE for a
comparison. After stating specific properties, the estimators are evaluated and
plotted. Make sure the datasets must be simulated before.
Created on Mon Dec 14 17:23:44 2020

@author: Franz Baumdicker, Klara Burger
"""
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys_path
from source.load_estimators import load_features
import matplotlib.font_manager as font_manager
from source.evaluate_estimators import evaluate_rho_variable

# evaluate differently trained NN (rho=0,35,1000 and rho variable), Watterson
# and ItMSE on a series of data sets with theta=40 and rho=0,1,2,3...:

# first state some characteristics about the series of data sets:
n = 40  # state sample size of the datasets
rep_test_data = 10000  # state number of repitions in each test dataset
rep_training_data = 200000  # state size of training dataset of NN for fixed rho
# state size of training dataset of NN trained for variable rho
rep_training_data_rho_var = 600000
rho = -1  # state recombination rate in each dataset
theta_min = 0  # minimal theta the NN were trained for
theta_max = 100  # maximal theta the NN were trained for
num_hidden_nodes = 200  # number of hidden nodes of the not linear NN
# set theta values
theta = 40.0
filepath = "../data/simulations/"  # state the filepath to the data sets

# evaluate estimators for variable recombination rate rho
evaluate_rho_variable(
    n,
    rep_test_data,
    theta,
    rho,
    num_hidden_nodes,
    rep_training_data,
    rep_training_data_rho_var,
    theta_min,
    theta_max,
    filepath,
    save=True,
)

# load estimators
NN_rho_0_nmse = load_features("NN_rho_0_nmse", n, rho, theta_min, theta_max)
NN_linear_rho_0_nmse = load_features(
    "Linear_NN_rho_0_nmse", n, rho, theta_min, theta_max
)
NN_rho_35_nmse = load_features("NN_rho_35_nmse", n, rho, theta_min, theta_max)
NN_linear_rho_35_nmse = load_features(
    "Linear_NN_rho_35_nmse", n, rho, theta_min, theta_max
)
NN_rho_1000_nmse = load_features("NN_rho_1000_nmse", n, rho, theta_min, theta_max)
NN_linear_rho_1000_nmse = load_features(
    "Linear_NN_rho_1000_nmse", n, rho, theta_min, theta_max
)
NN_rho_var_nmse = load_features("NN_rho_var_nmse", n, rho, theta_min, theta_max)
NN_linear_rho_var_nmse = load_features(
    "Linear_NN_rho_var_nmse", n, rho, theta_min, theta_max
)
Wat_nmse = load_features("Wat_nmse", n, rho, theta_min, theta_max)
ItMSE_nmse = load_features("ItMSE_nmse", n, rho, theta_min, theta_max)

# set figure size
fig = plt.figure(figsize=(15, 22.5))
plt.rcParams.update({"font.size": 24})
font = font_manager.FontProperties(family="Arial", size=24)

val = []
val = list(range(0, 51))
val.append(56)

# subplot 1
ax = plt.subplot(2, 1, 1)
# plot estimation mean for different estimators
(line1,) = plt.plot(
    val,
    NN_rho_0_nmse,
    "k+",
    color="tab:orange",
    label=r"NN, trained with $\rho=0$",
    linewidth=4,
)
(line2,) = plt.plot(
    val,
    NN_rho_35_nmse,
    "k*",
    color="tab:brown",
    label=r"NN, trained with $\rho=35$",
    linewidth=4,
)
(line3,) = plt.plot(
    val,
    NN_rho_1000_nmse,
    "k-.",
    color="tab:pink",
    label=r"NN, trained with $\rho=1000$",
    linewidth=4,
)
(line4,) = plt.plot(
    val,
    NN_rho_var_nmse,
    "k-",
    color="tab:red",
    label=r"Adaptive NN, trained with $\rho$ variable",
    linewidth=4,
)
(line5,) = plt.plot(
    val,
    NN_linear_rho_var_nmse,
    "k--",
    color="tab:blue",
    label=r"Linear NN, trained with $\rho$ variable",
    linewidth=4,
)
(line6,) = plt.plot(
    val, Wat_nmse, "k:", color="tab:green", label="Watterson", linewidth=4
)
(line7,) = plt.plot(
    val, ItMSE_nmse, "kx", color="tab:purple", label="ItMSE", linewidth=4
)

val2 = list(range(0, 57))
my_ticks = []
for i in range(0, 57):
    my_ticks.append(i)
    if i > 50:
        my_ticks[i] = ""
    if i == 53:
        my_ticks[i] = "..."
    if i == 56:
        my_ticks[i] = 1000

# set labels for x axis
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(
    val2,
    [
        "0",
        "",
        "",
        "",
        "",
        "5",
        "",
        "",
        "",
        "",
        "10",
        "",
        "",
        "",
        "",
        "15",
        "",
        "",
        "",
        "",
        "20",
        "",
        "",
        "",
        "",
        "25",
        "",
        "",
        "",
        "",
        "30",
        "",
        "",
        "",
        "",
        "35",
        "",
        "",
        "",
        "",
        "40",
        "",
        "",
        "",
        "",
        "45",
        "",
        "",
        "",
        "",
        "50",
        "",
        "",
        "...",
        "",
        "",
        "1000",
    ],
)

plt.ylabel("normalised MSE", fontname="Arial")
plt.xlabel(r"$\rho$", fontname="Arial")
plt.legend(loc="upper right", prop=font)
# plt.title(
#   r"Normalised MSE for $n=40$ and $\theta = 40$"
# )

# plot
plt.show()

# save plot as pdf
pp = PdfPages("n_40_theta_" + str(theta) + "_rho_0_50_test.pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
