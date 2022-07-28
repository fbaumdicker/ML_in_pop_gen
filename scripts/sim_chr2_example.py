#!/usr/bin/env python3

"""script to simulate data based on the recombination map of human chromosome 2."""

import numpy as np
import scipy.stats
import stdpopsim
import matplotlib.pyplot as pyplot
import collections

# specify genome of interest
species = stdpopsim.get_species("HomSap")
contig = species.get_contig("chr2", genetic_map="HapMapII_GRCh37")
model = stdpopsim.PiecewiseConstantSize(species.population_size)
mutation_rate = np.array(contig.mutation_rate)


# specify simulation properties
num_samples = 40
seed = 48
lenght_bin_sim = 7000

# simulate tree sequence
samples = model.get_samples(num_samples)
engine = stdpopsim.get_engine("msprime")
ts = engine.simulate(model, contig, samples, seed=seed)

# compute genotype matrix:
G = ts.genotype_matrix()
assert (G.shape[1] == num_samples)

# compute site frequency spectrum: sum over columns of the genotype matrix
a = G.sum(axis=1)
# global site frequency spectrum
S = np.zeros((num_samples - 1,), dtype=int)
for i in range(0, num_samples - 1):
    S[i] = collections.Counter(a)[i + 1]
# local SFS
mut_positions = np.empty(ts.num_mutations)
for i, variant in enumerate(ts.variants()):
    mut_positions[i] = variant.position

# compute SFS and genotype matrix along the genome
num_bins = int(contig.recombination_map.get_sequence_length()/lenght_bin_sim)
s, mut_bin_edges, _ = scipy.stats.binned_statistic(
    mut_positions, a, bins=num_bins, statistic='count')

upper = s.cumsum()
lower = np.insert(upper[:-1], 0, 0., axis=0)

SFS_along_chr = []
G_along_chr = []

for lo, up in zip(lower, upper):
    S = np.zeros((num_samples - 1,), dtype=int)
    for i in range(0, num_samples - 1):
        S[i] = collections.Counter(a[int(lo):int(up)])[i + 1]
    SFS_along_chr.append(S)
    G_along_chr.append(G[int(lo):int(up), ])

# compute recombination rate
positions = np.array(contig.recombination_map.get_positions()[1:])
rates = np.array(contig.recombination_map.get_rates()[1:])
v, bin_edges, _ = scipy.stats.binned_statistic(
    positions, rates, bins=num_bins)
v = v * ts.sequence_length / num_bins * species.population_size * 4

# plot recombination map
x = bin_edges[:-1][np.logical_not(np.isnan(v))]
y = v[np.logical_not(np.isnan(v))]
fig, ax1 = pyplot.subplots(figsize=(160, 6))
ax1.plot(x, y, color="blue")
ax1.set_ylabel("Recombination rate")
ax1.set_xlabel("Chromosome position")
fig.savefig("recomb_along_chr2")

# compute underlying mutation rate
true_theta = ts.sequence_length / num_bins * species.genome.mean_mutation_rate * species.population_size * 4
print("theta:", true_theta)

# save simulation
np.savez("../data/simulations/SFS_along_chr2_seed_" + str(seed), SFS_along_chr=SFS_along_chr, v=v,
         positions=positions, mut_bin_edges=mut_bin_edges, true_theta=true_theta, G_along_chr=G_along_chr)

print("simulation done.")