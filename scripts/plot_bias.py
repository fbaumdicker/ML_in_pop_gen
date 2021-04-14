"""
plot_bias.py: script to plot the normalised bias of Watterson, ItMSE, NN
and linear NN for recombination rates rho=0,35,1000. Note, the y scale is log
transformed.
Created on Mon Dec 14 17:23:44 2020

@author: Franz Baumdicker, Klara Burger
"""
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys_path
from source.load_estimators import load_features

# set characteristics of data used of evaluation
n = 40  # sample size
rho = 0  # recombination rate
theta_min = 0  # minimal theta value
theta_max = 40  # maximal theta value

# load nmse of estimators for rho=0:
Wat_mean = load_features("Wat_mean", n, rho, theta_min, theta_max)
ItMSE_mean = load_features("ItMSE_mean", n, rho, theta_min, theta_max)
NN_mean = load_features("NN_mean", n, rho, theta_min, theta_max)
Linear_NN_mean = load_features("Linear_NN_mean", n, rho, theta_min, theta_max)
val = load_features("val", n, rho, theta_min, theta_max)


# set figure size
fig = plt.figure(figsize=(10, 10))
plt.rcParams.update({"font.size": 24})


# plot
ax = plt.subplot(1, 1, 1)
# plot normalised MSE for different estimators for rho=0
(line1,) = plt.plot(val, NN_mean / val, "-", color="tab:red", label=r"Adaptive NN")
(line2,) = plt.plot(
    val, Linear_NN_mean / val, "-.", color="tab:blue", label=r"Linear NN"
)
(line3,) = plt.plot(val, Wat_mean / val, ":", color="tab:green", label="Watterson")
(line4,) = plt.plot(val, ItMSE_mean / val, "--", color="tab:purple", label="ItMSE")
plt.ylabel(r"mean $\theta$ estimate (normalised)")
plt.xlabel(r"$\theta$")
plt.title(r"$\rho=$" + str(rho))
plt.legend(loc="lower right")
plt.grid()

# plot
plt.show()

# save plot as pdf
pp = PdfPages("bias_n_40_rho_" + str(rho) + ".pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
