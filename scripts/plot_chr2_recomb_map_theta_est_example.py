""" script to plot recombination map for human chromosome 2 and theta
estimation of the adaptive neural network, linear neural network,
watterson's estimator and the iterative version of Futschik's and
Gach's estimator along the genome.

Note: SFS needs to be computed via the script sim_chr2.py before
running this script.
"""

import numpy
import stdpopsim
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as font_manager
import pandas as pd
from tensorflow import keras
from source.model_based_estimators import ItMSE, watterson
import msprime

# specify properties of SFS computation (see script sim_chr2.py).
num_sample = 40
seed = 48
true_theta = 36.12

# colourblind-friendly colour-palette (kudos to Masataka Okabe and Kei Ito).
black = numpy.array([[0.0, 0.0, 0.0]])
orange = numpy.array([[0.9, 0.62, 0.0]])
skyblue = numpy.array([[0.34, 0.71, 0.91]])
bluishgreen = numpy.array([[0.0, 0.62, 0.45]])
yellow = numpy.array([[0.94, 0.89, 0.26]])
blue = numpy.array([[0.0, 0.45, 0.7]])
vermilion = numpy.array([[0.84, 0.37, 0.0]])
reddishpurple = numpy.array([[0.8, 0.47, 0.65]])

# get recombination map for human chromosome 2 (chr2).
species = stdpopsim.get_species("HomSap")
contig = species.get_contig("chr2", genetic_map="HapMapII_GRCh37")
positions = numpy.array(contig.recombination_map.get_positions())
rates = numpy.array(contig.recombination_map.get_rates())

# load computed SFS for chr2.
df1 = []
mut_bin = []
input_data = numpy.load(
    "../data/simulations/SFS_along_chr2_seed_" + str(seed) + ".npz", allow_pickle=True
)
for SFS, mut in zip(input_data["SFS_along_chr"], input_data["mut_bin_edges"]):
    df1.append(SFS[0 : num_sample - 1].tolist())
    mut_bin.append(mut)

# compute SFS based on a sliding window of size 70kb for theta estimation.
print("compute SFS based on a sliding window of size 70kb for theta estimation.")
df2 = numpy.array(df1)
sum_SFS = numpy.zeros((1, num_sample - 1))[0]
moving_SFS = []
for i in range(0, len(df2) - 10):
    for j in range(0, 10):
        sum_SFS += df2[i + j]
    moving_SFS.append(sum_SFS)
    sum_SFS = numpy.zeros((1, num_sample - 1))[0]
df2 = numpy.array(moving_SFS)
X_test = pd.DataFrame(df2)

# compute recombination rates along the chr2 based on a sliding window of 70kb.
print(
    "compute recombination rates along the chr2 based on a sliding window of 70kb."
    "this step may take 3-5min, to reduce runtime save calculation result und load"
    "before next run."
)
rate_map = msprime.RateMap(position=positions, rate=rates[:-1])
moving_rho = []
for i in range(0, len(df1) - 10):
    sum_rho = 0
    moving_rho.append(
        numpy.sum(rate_map.get_rate(range(int(mut_bin[i]), int(mut_bin[i + 10]))))
        * species.population_size
        * 4
    )
moving_rho = numpy.array(moving_rho)
print("mean recombination rate of chr2:", numpy.mean(moving_rho))

# test estimators on chr2 data
# ANN
ann_pred = keras.models.load_model(
    "../data/saved_NN/adaptive_NN_1hl_200_n_40_rep_600000_rho_var_theta_random-100_1",
    compile=False,
).predict(X_test)
ann_pred_array = numpy.array([ann_pred[i, 0] for i in range(0, len(df2))])

# Linear NN
lin_nn_pred = keras.models.load_model(
    "../data/saved_NN/Linear_NN_n_40_rep_600000_rho_var_theta_random-100", compile=False
).predict(X_test)
lin_nn_pred_array = numpy.array([lin_nn_pred[i, 0] for i in range(0, len(df2))])

# Futschik (iter)
futschik_pred = ItMSE(df2, 1e-3)

# Watterson
est_watterson = []
for i in range(0, len(df2)):
    est_watterson.append(watterson(numpy.array(df2[i])))
wat_pred = numpy.array(est_watterson)

# create vector containing the underlying theta in each entry to plot as line later on.
theta_est = numpy.array(numpy.zeros(len(df2)))
theta_est.fill(true_theta)

# plot
# parameters for xlim and ylim
a = 0.01
b = 2.43
c = 400

# plot recombination rate
fig = plt.figure(figsize=(30, 10))
plt.rcParams.update({"font.size": 25})
font = font_manager.FontProperties(size=25)

ax1 = plt.subplot(1, 1, 1)
mut_bin_rho = mut_bin[4:-6]
(line1,) = plt.step(
    numpy.array(mut_bin_rho)[numpy.logical_not(numpy.isnan(moving_rho))],
    moving_rho[numpy.logical_not(numpy.isnan(moving_rho))],
    "tab:gray",
)
ax1.set_ylabel("Recombination rate")
ax1.set_xlabel("Chromosome position")
plt.xlim(a * 1e8, b * 1e8)
plt.ylim(-10, c)

# show and save plot as pdf
plt.show()
pp = PdfPages("recombination_map_along_chr2.pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
print("plot of recombination map created")


# plot theta estimation along the genome.
fig = plt.figure(figsize=(30, 10))
plt.rcParams.update({"font.size": 25})
font = font_manager.FontProperties(size=25)

ax2 = plt.subplot(1, 1, 1)
mut_bin_est = mut_bin[4:-6]
(line1,) = plt.plot(
    numpy.array(mut_bin_est)[numpy.logical_not(numpy.isnan(wat_pred))],
    wat_pred[numpy.logical_not(numpy.isnan(wat_pred))],
    color=bluishgreen,
    label=r"Watterson",
)
(line2,) = plt.plot(
    numpy.array(mut_bin_est)[numpy.logical_not(numpy.isnan(futschik_pred))],
    futschik_pred[numpy.logical_not(numpy.isnan(futschik_pred))],
    color=reddishpurple,
    label=r"Futschik (iter)",
)
(line3,) = plt.plot(
    numpy.array(mut_bin_est)[numpy.logical_not(numpy.isnan(lin_nn_pred_array))],
    lin_nn_pred_array[numpy.logical_not(numpy.isnan(lin_nn_pred_array))],
    color=vermilion,
    label=r"Linear NN",
)
(line4,) = plt.plot(
    numpy.array(mut_bin_est)[numpy.logical_not(numpy.isnan(ann_pred_array))],
    ann_pred_array[numpy.logical_not(numpy.isnan(ann_pred_array))],
    color=blue,
    label=r"Adaptive NN",
)
(line5,) = plt.plot(
    numpy.array(mut_bin_est)[numpy.logical_not(numpy.isnan(theta_est))],
    theta_est[numpy.logical_not(numpy.isnan(theta_est))],
    label=r"true $\theta$",
    color="tab:gray",
)
ax2.set_ylabel((r"$\theta$ estimate"))
ax2.set_xlabel("Chromosome position")
plt.legend(loc="upper left")
plt.xlim(a * 1e8, b * 1e8)

# show and save plot as pdf
plt.show()
pp = PdfPages("theta_est_along_chr2.pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
print("estimation plot created")
