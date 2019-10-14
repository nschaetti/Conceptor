# -*- coding: utf-8 -*-
#
# File : subspace_demo.py
# Description : Section 3.2 of the paper "Controlling RNN with Conceptors"
# Date : 14th of october, 2019
#
# This file is part of the Conceptor package.  The Conceptor package is free
# software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>
#

# Plain demo that when a RNN is driven by different signals, the induced
# internal signals will inhabit different subspaces of the signal space.

# Imports
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import math
from scipy.linalg import svd
import scipy.io as io
import argparse
from scipy.interpolate import interp1d
from patterns import patts
import reservoir
import measures
import conceptors
import tools


# Experiment control
seed = 1
reservoir_size = 100
spectral_radius = 1.5
input_scaling = 1.5
bias_scaling = 0.2
ridge_param_wstar = 0.0001
washout_length = 500
learn_length = 1000
signal_plot_length = 20
ridge_param_wout = 0.01
alpha = 10
conceptor_test_length = 200
singular_plot_length = 50
free_run_length = 100000
alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]

# Setting patterns
patterns = [52, 53, 9, 35]

# Initialise random generator
np.random.seed(seed)

# Argument parsing
parser = argparse.ArgumentParser(prog="subspace_demo", description=u"Fig. 1 BC subspace first demo")
parser.add_argument("--w", type=str, default="", required=False)
parser.add_argument("--win", type=str, default="", required=False)
parser.add_argument("--wbias", type=str, default="", required=False)
args = parser.parse_args()

# Compute connectivity
if reservoir_size <= 20:
    connectivity = 1.0
else:
    connectivity = 10.0 / reservoir_size
# end if

# Load W from matlab file and random init ?
if args.w != "":
    # Load internal weights
    Wstar_raw = reservoir.load_matlab_file(args.w, args.w_name)
else:
    # Generate internal weights
    Wstar_raw = reservoir.generate_internal_weights(reservoir_size, connectivity, seed)
# end if

# Load Win from matlab file or init randomly
if args.win != "":
    Win_raw = reservoir.load_matlab_file(args.win, args.win_name)
else:
    # Generate Win
    Win_raw = np.random.randn(reservoir_size, 1)
# end if

# Load Wbias from matlab from or init randomly
if args.wbias != "":
    Wbias_raw = np.random.randn(reservoir_size)
else:
    Wbias_raw = io.loadmat("data/params/WbiasRaw.mat")['WbiasRaw'].reshape(-1)
# end if

# Scale raw weights and initialize weights
Wstar, Win, Wbias = reservoir.scale_weights(
    W=Wstar_raw,
    Win=Win_raw,
    Wbias=Wbias_raw,
    spectral_radius=spectral_radius,
    input_scaling=input_scaling,
    bias_scaling=bias_scaling
)

# Identity matrix
I = np.eye(reservoir_size)

# Number of patterns
n_patterns = len(patterns)

# We will save all states
training_states = np.zeros((reservoir_size, n_patterns * learn_length))
training_states_old = np.zeros((reservoir_size, n_patterns * learn_length))
training_win_u = np.zeros((reservoir_size, n_patterns * learn_length))
training_targets = np.zeros((1, n_patterns * learn_length))

# Readout matrix
readout_weights = np.empty([1, n_patterns])

# Save all patterns
pattern_collectors = np.zeros((n_patterns, learn_length))

# Save states
state_collectors_centered = np.zeros((n_patterns, reservoir_size, learn_length))
state_collectors = np.zeros((n_patterns, reservoir_size, learn_length))

# Save singular values
singular_value_collectors = np.zeros((n_patterns, reservoir_size))

# Save output vectors
unitary_matrix_collectors = np.zeros((n_patterns, reservoir_size, reservoir_size))

# Save all coRrelation matrices
correlation_matrix_collectors = np.zeros((n_patterns, reservoir_size, reservoir_size))

# State and pattern for plotting
plotting_states = np.zeros((n_patterns, reservoir_size, signal_plot_length))
plotting_patterns = np.zeros((n_patterns, signal_plot_length))

# Last states
last_states = np.zeros((reservoir_size, n_patterns))

# Collection of conceptors
conceptor_collector = conceptors.Collector(reservoir_size)

# We run the ESN model and we save
# all the needed informations.
# For each patterns
for p in range(n_patterns):
    # Select the right pattern
    patt = patts[patterns[p]]

    # Starting state (x_0 = 0)
    x = np.zeros(reservoir_size)

    # Run reservoir
    state_collector, pattern_collector = reservoir.run(
        pattern=patt,
        reservoir_size=reservoir_size,
        x_start=x,
        Wstar=Wstar,
        Win=Win,
        Wbias=Wbias,
        training_length=learn_length,
        washout_length=washout_length
    )

    # State collector
    state_collectors[p] = state_collector

    # Time shifted states
    state_collector_old = np.zeros((reservoir_size, learn_length))
    state_collector_old[:, 1:] = state_collector[:-1]

    # Save last state
    last_states[:, p] = state_collector[:, -1]

    # Train conceptor
    C, U, Snew, Sorg = conceptors.train(
        X=state_collector,
        aperture=alpha,
        dim=1
    )

    # Save conceptor
    conceptor_collector.add(p, C, U, Snew, Sorg)

    # Plotting states and patterns
    plotting_states[p] = state_collector[:, :signal_plot_length]
    plotting_patterns[p] = pattern_collector[:signal_plot_length]

    # Save patterns
    pattern_collectors[p] = pattern_collector

    # Save states for training
    training_states[:, p * learn_length:(p + 1) * learn_length] = state_collector
    training_states_old[:, p * learn_length:(p + 1) * learn_length] = state_collector_old
    training_targets[0, p * learn_length:(p + 1) * learn_length] = pattern_collector
    training_win_u[:, p * learn_length:(p + 1) * learn_length] = Win * pattern_collector
# end for

# Compute readout
Wout = reservoir.train_outputs(
    training_states=training_states,
    training_targets=training_targets,
    ridge_param_wout=ridge_param_wout,
    dim=1
)

# Compute training error (NRMSE)
NRMSE_readout = measures.nrmse(np.dot(Wout, training_states), training_targets)
print(u"NRMSE readout: {}".format(NRMSE_readout))

# Load W
W, X_new = reservoir.loading(
    X=training_states,
    Xold=training_states_old,
    Wbias=Wbias,
    ridge_param=ridge_param_wstar,
    dim=1
)

# Training errors per neuron
NRMSE_W = measures.nrmse(np.dot(W, training_states_old), X_new)
print(u"mean NRMSE W: {}".format(np.average(NRMSE_W)))

# Run loaded reservoir to observe a messy output. Do this with starting
# from four states originally obtained in the four driving conditions
# initialize network state.
for p in range(4):
    # Run the loaded reservoir
    free_run_states, free_run_outputs = reservoir.free_run(
        x_start=last_states[:, p],
        W=W,
        Wbias=Wbias,
        Wout=Wout,
        C=None,
        run_length=conceptor_test_length,
        washout_length=0
    )
# end for

# Now we generate signal but with conceptors
# Save states and outputs
for p in range(n_patterns):
    # Get the corresponding conceptors
    C, _, _, _ = conceptor_collector.get(p)

    # Original starting state or get the original
    x = 0.5 * np.random.randn(reservoir_size)

    # Free run with the loaded reservoir
    conceptor_test_states, conceptor_test_output = reservoir.free_run(
        x_start=x,
        W=W,
        Wbias=Wbias,
        Wout=Wout,
        C=C,
        run_length=conceptor_test_length,
        washout_length=0
    )
# end for

# Variables for plotting
interpolation_rate = 20

# Aligned results
conceptor_test_output_aligned = np.zeros((n_patterns, signal_plot_length))
conceptor_test_states_aligned = np.zeros((n_patterns, signal_plot_length, 5))

# Measured performances
NRMSE_aligned = np.zeros(n_patterns)
MSE_aligned = np.zeros(n_patterns)

# For each pattern
for p in range(n_patterns):
    # Interpolate and align truth and generated
    aligned_ouputs, max_ind = tools.interpolation_alignment(
        truth_pattern=plotting_patterns[p],
        generated_pattern=conceptor_test_output[p],
        interpolation_rate=interpolation_rate,
        dim=1
    )

    # Get aligned states
    conceptor_test_states_aligned[p] = conceptor_test_states[p, max_ind:max_ind + signal_plot_length]

    # Save aligned outputs
    conceptor_test_output[p] = aligned_ouputs

    # Evaluate aligned signals with NRMSE
    NRMSE_aligned[p] = measures.nrmse(
        conceptor_test_output_aligned[p].reshape(1, -1),
        plotting_patterns[p].reshape(1, -1)
    )

    # Evaluate aligned signals with MSE
    MSE_aligned[p] = np.average(np.power(conceptor_test_output_aligned[p] - plotting_patterns[p], 2))
# end for

# Show NRMSE and MSE and their average
print(u"NRMSE aligned : {}".format(NRMSE_aligned))
print(u"MSE aligned : {}".format(MSE_aligned))
print(u"Average MSE : {}".format(np.average(NRMSE_aligned)))

# Figure (square size)
plt.figure(figsize=(12, 8))

# For each pattern
for p in range(n_patterns):
    # Plot 1 : original pattern and recreated pattern
    plt.subplot(n_patterns, 4, p * 4 + 1)
    plt.plot(conceptor_test_output_aligned[p], color='r', linewidth=5)
    plt.plot(plotting_patterns[p], color='b', linewidth=1.5)

    # Title
    if p == 0:
        plt.title(u'p and y')
    # end if

    # X labels
    if p == 3:
        plt.xticks([0, 10, 20])
    else:
        plt.xticks([])
    # end if

    # Y limits
    plt.ylim([-1, 1])
    plt.yticks([-1, 0, 1])

    # Plot 2 : states
    plt.subplot(n_patterns, 4, p * 4 + 2)
    plt.plot(plotting_states[p, 0])
    plt.plot(plotting_states[p, 1])
    plt.plot(plotting_states[p, 2])

    # Title
    if p == 0:
        plt.title(u'two neurons')
    # end if

    # X labels
    if p == 3:
        plt.xticks([0, 10, 20])
    else:
        plt.xticks([])
    # end if

    # Y limits
    plt.ylim([-1, 1])
    plt.yticks([-1, 0, 1])

    # Plot 3 : Log10 of singular values (PC energy)
    plt.subplot(n_patterns, 4, p * 4 + 3)
    plt.plot(np.log10(singular_value_collectors[p]), 'red', linewidth=2)

    # X labels
    if p == 3:
        plt.xticks([0, 50, 100])
    else:
        plt.xticks([])
    # end if

    # Limits
    plt.ylim([-20, 10])

    # Title
    if p == 0:
        plt.title(u'log10 PC Energy')
    # end if

    # Learning PC energy
    plt.subplot(n_patterns, 4, p * 4 + 4)
    plt.plot(singular_value_collectors[p, :10], 'red', linewidth=2)

    # Title
    if p == 0:
        plt.title(u"leading PC energy")
    # end if

    # Limits
    plt.ylim([0, 40.0])
# end for

# Show figure
plt.show()

# Energy similarities between driven response spaces

# List of conceptor similarities
similarity_matrix_conceptors = np.zeros((n_patterns, n_patterns))

# For each combination of patterns
for p1 in range(n_patterns):
    for p2 in range(n_patterns):
        similarity_num = math.pow(
            lin.norm(
                np.dot(
                    np.dot(
                        np.dot(
                            np.diag(np.sqrt(Cs[p1][2])),
                            Cs[p1][1].T
                        ),
                        Cs[p2][1]
                    ),
                    np.diag(np.sqrt(Cs[p2][2]))
                ),
                'fro'),
            2)

        # Div
        similarity_div = np.dot(
            lin.norm(Cs[p1][0], 'fro'),
            lin.norm(Cs[p2][0], 'fro')
        )

        # Similarity
        similarity = similarity_num / similarity_div

        # Save
        similarity_matrix_conceptors[p1, p2] = similarity
    # end for
# end for

# List of correlation matrix similarities
similarity_matrix_correlations = np.zeros((n_patterns, n_patterns))

# For each combination of patterns
for p1 in range(n_patterns):
    for p2 in range(n_patterns):
        similarity_num = math.pow(
            lin.norm(
                np.dot(
                    np.dot(
                        np.dot(
                            np.sqrt(np.diag(singular_value_collectors[p1])),
                            unitary_matrix_collectors[p1].T
                        ),
                        unitary_matrix_collectors[p2]
                    ),
                    np.sqrt(np.diag(singular_value_collectors[p2]))
                ),
                'fro'),
            2)

        # Div
        similarity_div = np.dot(
            lin.norm(correlation_matrix_collectors[p1], 'fro'),
            lin.norm(correlation_matrix_collectors[p2], 'fro')
        )

        # Similarity
        similarity = similarity_num / similarity_div

        # Save
        similarity_matrix_correlations[p1, p2] = similarity
    # end for
# end for

# Figure (square size)
# fig = plt.figure(figsize=(5, 5))

# Show similarity matrices
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(similarity_matrix_conceptors, interpolation='nearest', cmap='Greys_r')
plt.title(u"C based similaritites, a = 10")
fig.colorbar(cax, ticks=np.arange(0.1, 1.1, 0.1))
for (i, j), z in np.ndenumerate(similarity_matrix_conceptors):
    if (i < 2 and j < 2) or (i > 1 and j > 1):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    else:
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color='white')
    # end if
# end for
plt.show()

# Figure (square size)
# plt.figure(figsize=(5, 5))

# Show R matrix
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(similarity_matrix_correlations, interpolation='nearest', cmap='Greys_r')
plt.title(u"R based similaritites")
fig.colorbar(cax, ticks=np.arange(0.1, 1.1, 0.1))
for (i, j), z in np.ndenumerate(similarity_matrix_correlations):
    if (i < 2 and j < 2) or (i > 1 and j > 1):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    else:
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color='white')
    # end if
# end for
plt.show()

# Plotting comparisons for different alphas

# Save each singular values
plotting_singular_values_sine = np.zeros((len(alphas), reservoir_size))
plotting_singular_values_periodic = np.zeros((len(alphas), reservoir_size))

# For each alpha value
for i, a in enumerate(alphas):
    # Sine
    R1 = correlation_matrix_collectors[0]
    C1 = np.dot(R1, lin.inv(R1 + np.power(a, -2) * I))
    U1, S1, V1 = lin.svd(C1)
    plotting_singular_values_sine[i] = S1

    # Periodic
    R2 = correlation_matrix_collectors[2]
    C2 = np.dot(R2, lin.inv(R2 + np.power(a, -2) * I))
    U2, S2, V2 = lin.svd(C2)
    plotting_singular_values_periodic[i] = S2
# end for

# Figure (square size)
plt.figure(figsize=(12, 4))

# Plot sine singular values
plt.subplot(1, 2, 1)
for i in range(len(alphas)):
    plt.plot(plotting_singular_values_sine[i], linewidth=2)
# end for
plt.title(u"Sine (pattern 1)")
plt.yticks([0, 1])
plt.ylim([0, 1.1])

# Plot periodic
plt.subplot(1, 2, 2)
for i in range(len(alphas)):
    plt.plot(plotting_singular_values_periodic[i], linewidth=2)
# end for
plt.yticks([0, 1])
plt.ylim([0, 1.1])
plt.title(u"10-periodic random (pattern 1)")

# Show
plt.show()
