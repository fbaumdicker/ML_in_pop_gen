""" script to create violin plots for simulated data from recombination map for human chromosome 2. adaptive neural
network, linear neural network, watterson's estimator and the iterative version of Futschik's and Gach's estimator are
considered.

Note: SFS needs to be computed via the script sim_chr2.py before running this script.
"""

import numpy
import stdpopsim
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import math
from tensorflow import keras
import matplotlib.patches as mpatches
from source.model_based_estimators import ItMSE, watterson
import seaborn as sns
import msprime
import string

# specify properties of SFS computation (see script sim_chr2.py).
num_sample = 40
seed = 48
true_theta = 36.12

# specify ranges for recombination rates for which violin plots should be created. in this case 3 classes: 0<rho<=1, 30<rho<=40 and rho>150.
rho_classes = [0,1,30,40,150]
num_classes = 3

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
contig = species.get_contig("chr2", genetic_map = "HapMapII_GRCh37")
positions = numpy.array(contig.recombination_map.get_positions())
rates = numpy.array(contig.recombination_map.get_rates()[:-1])

# load computed SFS for chr2
df1 = []
mut_bin = []
input_data = numpy.load("../data/simulations/SFS_along_chr2_seed_"+str(seed)+".npz", allow_pickle=True)
for SFS, mut in zip(input_data["SFS_along_chr"], input_data["mut_bin_edges"]):
    df1.append(SFS[0 : num_sample - 1].tolist())
    mut_bin.append(mut)

rate_map = msprime.RateMap(position=positions, rate=rates)
moving_rho = []

df2 = numpy.array(df1)
sum_SFS = numpy.zeros((1, num_sample-1))[0]
sum_rho = 0

# sort SFS into 3 classes by mean recombination rates of disjoint 70kb windows of chr2.
moving_rho_sorted = [[] for i in range(0, 3)]
moving_SFS_sorted = [[] for i in range(0, 3)]
for i in range(0, math.floor(len(df1)/10)):
        # compute mean recombination rate in each of the disjoint 70kb windows
        sum_rho = numpy.sum(rate_map.get_rate(range(int(mut_bin[i*10]), int(mut_bin[(i*10)+10])))) * species.population_size * 4
        # get SFS for each window
        for j in range(0, 10):
            sum_SFS += df2[(i*10) + j]
        # sort SFS into appropriate class
        if rho_classes[0] < sum_rho <= rho_classes[1]:
                moving_rho_sorted[0].append(sum_rho)
                moving_SFS_sorted[0].append(sum_SFS)
        if rho_classes[2] < sum_rho <= rho_classes[3]:
                moving_rho_sorted[1].append(sum_rho)
                moving_SFS_sorted[1].append(sum_SFS)
        if sum_rho > rho_classes[4]:
                moving_rho_sorted[2].append(sum_rho)
                moving_SFS_sorted[2].append(sum_SFS)
        sum_SFS = numpy.zeros((1, num_sample - 1))[0]
        sum_rho = 0

# apply estimators to SFS for each class.
ann_pred_classes = []
lin_nn_pred_classes = []
wat_pred_classes = []
futschik_pred_classes = []

for i in range(0, num_classes):
    df2 = numpy.array(moving_SFS_sorted[i])
    X_test = pd.DataFrame(df2)
    ann_pred = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_200_n_" + str(num_sample) + "_rep_600000_rho_var_theta_random-100_1", compile=False).predict(X_test)
    ann_pred_classes.append(numpy.array([ann_pred[j, 0] for j in range(0, len(df2))]))
    lin_nn_pred = keras.models.load_model("../data/saved_NN/Linear_NN_n_40_rep_600000_rho_var_theta_random-100", compile=False).predict(X_test)
    lin_nn_pred_classes.append(numpy.array([lin_nn_pred[j, 0] for j in range(0, len(df2))]))
    futschik_pred_classes.append(ItMSE(df2, 1e-3))
    est_watterson = []
    for j in range(0, len(df2)):
        est_watterson.append(watterson(numpy.array(df2[j])))
    wat_pred_classes.append(numpy.array(est_watterson))


# combine different predictions into a list for each rho class.
data1 = []
data2 =[]
data3 = []
data1.append(ann_pred_classes[0])
data1.append(lin_nn_pred_classes[0])
data1.append(wat_pred_classes[0])
data1.append(futschik_pred_classes[0])
data2.append(ann_pred_classes[1])
data2.append(lin_nn_pred_classes[1])
data2.append(wat_pred_classes[1])
data2.append(futschik_pred_classes[1])
data3.append(ann_pred_classes[2])
data3.append(lin_nn_pred_classes[2])
data3.append(wat_pred_classes[2])
data3.append(futschik_pred_classes[2])


## create violin plot
# compute whiskers
q1, q3 = numpy.percentile(data2, [25, 75])
whisker_low = q1 - (q3 - q1) * 1.5
whisker_high = q3 + (q3 - q1) * 1.5

# lighten the color grey
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
my_grey = lighten_color('tab:grey', 1.5)

# set figure properties, figure consists of 3 subplots.
fs = 1.5
sns.set(font_scale = fs)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 6))
axs = plt.subplot(1, 3, 1)
flierprops = dict(markerfacecolor='w', markersize=5,linestyle='none')

# set color style.
my_color = [blue, vermilion, bluishgreen, reddishpurple]

# violin plot for 0<rho<=1.
violin_plot1 = sns.violinplot(data=data1, palette=my_color, inner=None, saturation=0.6, cut=0)
sns.boxplot(data=data1, palette=my_color, width=0.035, boxprops={'facecolor':my_grey, 'color':my_grey, 'zorder': 2}, ax=violin_plot1, flierprops=flierprops,
            whiskerprops=dict(color=my_grey, linewidth=1.5), medianprops=dict(color="w", linewidth=2))
axs.hlines(true_theta, xmin=-0.5, xmax=3.5, color='tab:gray', lw=2)
axs.text(-0.085, 0.95, string.ascii_uppercase[0], transform=axs.transAxes, size=22, weight='bold')
violin_plot1.set_xticks([1.5])
violin_plot1.set_xticklabels([r'$0 <\rho \leq 1$'])

# add legend to subplot 1.
red_patch1 = mpatches.Patch(color=blue)
red_patch2 = mpatches.Patch(color=vermilion)
red_patch3 = mpatches.Patch(color=bluishgreen)
red_patch4 = mpatches.Patch(color=reddishpurple)
red_patch5 = mpatches.Patch(color='tab:gray')
label = ['Adaptive NN','Linear NN','Watterson','Futschik (iter)', r'true $\theta$']
fake_handles = [red_patch1, red_patch2, red_patch3, red_patch4, red_patch5]
violin_plot1.legend(fake_handles, label, loc="upper left", fontsize = 12)

# violin plot for 30<rho<=40.
axs = plt.subplot(1, 3, 2)
violin_plot2 = sns.violinplot(data=data2, palette=my_color,saturation=0.6, inner=None, cut=0)
sns.boxplot(data=data2, palette=my_color, width=0.035, boxprops={'facecolor':my_grey, 'color':my_grey, 'zorder': 2}, ax=violin_plot2, flierprops=flierprops,
            whiskerprops=dict(color=my_grey, linewidth=1.5), medianprops=dict(color="w", linewidth=2))
axs.hlines(true_theta,  xmin=-0.5, xmax=3.5,  color='tab:gray', lw=2)
axs.text(-0.09, 0.95, string.ascii_uppercase[1], transform=axs.transAxes, size=22, weight='bold')
violin_plot2.set_xticks([1.5])
violin_plot2.set_xticklabels([r'$30 <\rho \leq 40$'])

# violin plot for rho>150.
axs = plt.subplot(1, 3, 3)
violin_plot3 = sns.violinplot(data=data3, palette=my_color, saturation=0.6,inner=None, cut=0)
sns.boxplot(data=data3, palette=my_color, width=0.035, boxprops={'facecolor':my_grey, 'color':my_grey, 'zorder': 2}, ax=violin_plot3, flierprops=flierprops,
            whiskerprops=dict(color=my_grey, linewidth=1.5), medianprops=dict(color="w", linewidth=2))
axs.hlines(true_theta,  xmin=-0.5, xmax=3.5,  color='tab:gray', lw=2)
axs.text(-0.09, 0.95, string.ascii_uppercase[2], transform=axs.transAxes, size=22, weight='bold')
violin_plot3.set_xticks([1.5])
violin_plot3.set_xticklabels([r'$\rho > 150 $'])

# create and save violin plot.
plt.tight_layout()
plt.show()
pp = PdfPages("violinplot_chr2.pdf")
pp.savefig(fig, bbox_inches="tight")
pp.close()
print("plot created.")