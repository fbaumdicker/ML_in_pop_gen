"""
plot_nmse.py: script to plot the normalised MSE of Watterson, ItMSE, NN
and linear NN for fixed  recombination rates.
Created on Mon Dec 14 17:23:44 2020

@author: Franz Baumdicker, Klara Burger
"""

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as font_manager
import sys_path
from source.load_estimators import load_features


# set characteristics of data used of evaluation
n = 40  # sample size
rho = 0  # recombination rate
theta_min = 0  # minimal theta value
theta_max = 40  # maximal theta value

# load nmse of estimators
Wat_nmse = load_features("Wat_nmse", n, rho, theta_min, theta_max)
ItMSE_nmse = load_features("ItMSE_nmse", n, rho, theta_min, theta_max)
NN_nmse = load_features("NN_nmse", n, rho, theta_min, theta_max)
Linear_NN_nmse = load_features("Linear_NN_nmse", n, rho, theta_min, theta_max)
val = load_features("val", n, rho, theta_min, theta_max)

# set figure size
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update({"font.size": 30})
font = font_manager.FontProperties(size=30)

# plot normalised MSE for different estimators
ax = plt.subplot(1, 1, 1)
(line1,) = plt.plot(
    val, NN_nmse, "-", color="tab:red", label=r"Adaptive NN", linewidth=4
)
(line2,) = plt.plot(
    val, Linear_NN_nmse, "-.", color="tab:blue", label=r"Linear NN", linewidth=4
)
(line3,) = plt.plot(
    val, Wat_nmse, ":", color="tab:green", label="Watterson", linewidth=4
)
(line4,) = plt.plot(
    val, ItMSE_nmse, "--", color="tab:purple", label="ItMSE", linewidth=4
)
plt.ylabel("normalised MSE")
plt.xlabel(r"$\theta$")
plt.title(r"$\rho=$" + str(rho))
if rho == 0:
    plt.grid()
    ax.set_yticks(
        [
            0,
            0.25,
            0.5,
            0.75,
            1,
            1.25,
            1.5,
            1.75,
            2,
            2.25,
            2.5,
            2.75,
            3,
            3.25,
            3.5,
            3.75,
            4,
        ],
        minor=False,
    )
    ax.yaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="minor")
    plt.legend(loc="upper left", prop=font)
elif rho == 35:
    plt.grid()
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_yticks([-0.01, 0.25], minor=True)
    ax.yaxis.grid(True, which="major")
    ax.yaxis.grid(False, which="minor")
    plt.legend(loc="lower right", prop=font)
elif rho == 1000:
    plt.grid()
    ax.set_yticks([int(0), 0.25, 0.50], minor=False)
    ax.set_yticks([-0.01, 0.25], minor=True)
    ax.yaxis.grid(True, which="major")
    ax.yaxis.grid(False, which="minor")
    locs, labels = plt.yticks()
    plt.yticks([0, 0.25, 0.5], ["0.00", "0.25", "0.50"])
    plt.legend(loc="lower right", prop=font)
else:
    plt.grid()
    plt.legend(loc="lower right", prop=font)

# plot
plt.show()

# save plot as pdf
pp = PdfPages("nmse_n_40_rho_" + str(rho) + ".pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
