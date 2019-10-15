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
import scipy.io as io
import argparse

# Conceptors
from Conceptors.patterns import patts
import Conceptors.reservoir
import Conceptors.measures
import Conceptors.conceptors
import Conceptors.tools
import Conceptors.visualization
import Conceptors.logic


# Experiment control
seed = 1
reservoir_size = 100

# Weights scaling
spectral_radius = 1.5
input_scaling = 1.5
bias_scaling = 0.2

# Sequence lengths
washout_length = 500
learn_length = 1000
signal_plot_length = 20
conceptor_test_length = 200
singular_plot_length = 50
free_run_length = 100000

# Regularization
ridge_param_wstar = 0.0001
ridge_param_wout = 0.01

# Aperture
alpha = 10

# Apertures to run
alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]

# Setting patterns
patterns = [52, 53, 9, 35]

# Initialise random generator
np.random.seed(seed)

# Argument parsing
parser = argparse.ArgumentParser(prog="subspace_demo", description=u"Fig. 1 BC subspace first demo")
parser.add_argument("--w", type=str, default="", required=False)
parser.add_argument("--w-name", type=str, default="", required=False)
parser.add_argument("--win", type=str, default="", required=False)
parser.add_argument("--win-name", type=str, default="", required=False)
parser.add_argument("--wbias", type=str, default="", required=False)
parser.add_argument("--wbias-name", type=str, default="", required=False)
parser.add_argument("--x0", type=str, default="", required=False)
parser.add_argument("--x0-name", type=str, default="", required=False)
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
    Wstar_raw = Conceptors.reservoir.load_matlab_file(args.w, args.w_name)
else:
    # Generate internal weights
    Wstar_raw = Conceptors.reservoir.generate_internal_weights(reservoir_size, connectivity, seed)
# end if

# Load Win from matlab file or init randomly
if args.win != "":
    Win_raw = Conceptors.reservoir.load_matlab_file(args.win, args.win_name)
else:
    # Generate Win
    Win_raw = np.random.randn(reservoir_size, 1)
# end if

# Load Wbias from matlab from or init randomly
if args.wbias != "":
    Wbias_raw = Conceptors.reservoir.load_matlab_file(args.wbias, args.wbias_name).reshape(-1)
else:
    Wbias_raw = np.random.randn(reservoir_size)
# end if

# Scale raw weights and initialize weights
Wstar, Win, Wbias = Conceptors.reservoir.scale_weights(
    W=Wstar_raw,
    Win=Win_raw,
    Wbias=Wbias_raw,
    spectral_radius=spectral_radius,
    input_scaling=input_scaling,
    bias_scaling=bias_scaling
)

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
conceptor_collector = Conceptors.conceptors.Collector(reservoir_size)

# We run the ESN model and we save
# all the needed informations.
# For each patterns
for p in range(n_patterns):
    # Select the right pattern
    patt = patts[patterns[p]]

    # Starting state (x_0 = 0)
    x = np.zeros(reservoir_size)

    # Run reservoir
    state_collector, pattern_collector = Conceptors.reservoir.run(
        pattern=patt,
        reservoir_size=reservoir_size,
        x_start=x,
        Wstar=Wstar,
        Win=Win,
        Wbias=Wbias,
        run_length=learn_length,
        washout_length=washout_length
    )

    # State collector
    state_collectors[p] = state_collector

    # Time shifted states
    state_collector_old = np.zeros((reservoir_size, learn_length))
    state_collector_old[:, 1:] = state_collector[:, :-1]

    # Save last state
    last_states[:, p] = state_collector[:, -1]

    # Train conceptor
    C, U, Snew, Sorg, R = Conceptors.conceptors.train(
        X=state_collector,
        aperture=alpha
    )

    # Save conceptor and singular values
    conceptor_collector.add(p, C, U, Snew, Sorg, R)
    singular_value_collectors[p] = Sorg

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
Wout = Conceptors.reservoir.train_outputs(
    training_states=training_states,
    training_targets=training_targets,
    ridge_param_wout=ridge_param_wout
)

# Compute training error (NRMSE)
NRMSE_readout = Conceptors.measures.nrmse(np.dot(Wout, training_states), training_targets)
print(u"NRMSE readout: {}".format(NRMSE_readout))

# Load W
W, X_new = Conceptors.reservoir.loading(
    X=training_states,
    Xold=training_states_old,
    Wbias=Wbias,
    ridge_param=ridge_param_wstar
)

# Training errors per neuron
NRMSE_W = Conceptors.measures.nrmse(np.dot(W, training_states_old), X_new)
print(u"mean NRMSE W: {}".format(np.average(NRMSE_W)))

# Run loaded reservoir to observe a messy output. Do this with starting
# from four states originally obtained in the four driving conditions
# initialize network state.
for p in range(4):
    # Run the loaded reservoir
    free_run_states, free_run_outputs = Conceptors.reservoir.free_run(
        x_start=last_states[:, p],
        W=W,
        Wbias=Wbias,
        Wout=Wout,
        C=None,
        run_length=conceptor_test_length,
        washout_length=0
    )
# end for

# Output states and signals
conceptor_test_states = np.zeros((n_patterns, reservoir_size, conceptor_test_length))
conceptor_test_output = np.zeros((n_patterns, conceptor_test_length))

# Now we generate signal but with conceptors
# Save states and outputs
for p in range(n_patterns):
    # Get the corresponding conceptors
    C, _, _, _, _ = conceptor_collector.get(p)

    # Original starting state or get the original
    if args.x0 != "":
        original_x = Conceptors.reservoir.load_matlab_file("{}{}.mat".format(args.x0, p + 1), args.x0_name)
        x = original_x.reshape(reservoir_size)
    else:
        x = 0.5 * np.random.randn(reservoir_size)
    # end if

    # Free run with the loaded reservoir
    test_states, test_output = Conceptors.reservoir.free_run(
        x_start=x,
        W=W,
        Wbias=Wbias,
        Wout=Wout,
        C=C,
        run_length=conceptor_test_length,
        washout_length=0
    )

    # Save
    conceptor_test_states[p] = test_states
    conceptor_test_output[p] = test_output
# end for

# Variables for plotting
interpolation_rate = 20

# Aligned results
conceptor_test_output_aligned = np.zeros((n_patterns, signal_plot_length))
conceptor_test_states_aligned = np.zeros((n_patterns, 5, signal_plot_length))

# Measured performances
NRMSE_aligned = np.zeros(n_patterns)
MSE_aligned = np.zeros(n_patterns)

# For each pattern
for p in range(n_patterns):
    # Interpolate and align truth and generated
    aligned_ouputs, max_ind = Conceptors.tools.interpolation_alignment_1d(
        truth_pattern=plotting_patterns[p],
        generated_pattern=conceptor_test_output[p],
        interpolation_rate=interpolation_rate
    )

    # Get aligned states
    conceptor_test_states_aligned[p] = conceptor_test_states[p, :5, max_ind:max_ind + signal_plot_length]

    # Save aligned outputs
    conceptor_test_output_aligned[p] = aligned_ouputs

    # Evaluate aligned signals with NRMSE
    NRMSE_aligned[p] = Conceptors.measures.nrmse(
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

# Show patterns and singular values
Conceptors.visualization.plot_patterns_with_singular_values(
    truth_patterns=plotting_patterns,
    generated_patterns=conceptor_test_output_aligned,
    Xs=plotting_states,
    SVs=singular_value_collectors
)

# Energy similarities between driven response spaces
# List of conceptor similarities
similarity_matrix_conceptors = np.zeros((n_patterns, n_patterns))

# For each combination of patterns
for p1 in range(n_patterns):
    for p2 in range(n_patterns):
        # Get conceptors
        C1, U1, S1, _, _ = conceptor_collector.get(p1)
        C2, U2, S2, _, _ = conceptor_collector.get(p2)

        # Cosine similarity on conceptor matrices
        similarity = Conceptors.measures.conceptor_cosine_similarity(
            C1=C1,
            U1=U1,
            S1=S1,
            C2=C2,
            U2=U2,
            S2=S2
        )

        # Save
        similarity_matrix_conceptors[p1, p2] = similarity
    # end for
# end for

# List of correlation matrix similarities
similarity_matrix_correlations = np.zeros((n_patterns, n_patterns))

# For each combination of patterns
for p1 in range(n_patterns):
    for p2 in range(n_patterns):
        # Get conceptors
        _, U1, _, S1, R1 = conceptor_collector.get(p1)
        _, U2, _, S2, R2 = conceptor_collector.get(p2)

        # Cosine similarity on correlation matrix
        similarity = Conceptors.measures.generalized_cosine_similarity(
            R1=R1,
            U1=U1,
            S1=S1,
            R2=R2,
            U2=U2,
            S2=S2
        )

        # Save
        similarity_matrix_correlations[p1, p2] = similarity
    # end for
# end for

# Plot similarity matrix (conceptors)
Conceptors.visualization.plot_similarity_matrix(
    similarity_matrix=similarity_matrix_conceptors,
    title="C based similarities, a = 10"
)

# Plot similarity matrix (correlation matrix)
Conceptors.visualization.plot_similarity_matrix(
    similarity_matrix=similarity_matrix_correlations,
    title="R based similarities"
)

#
# Plotting comparisons for different alphas
#

# Save each singular values
plotting_singular_values_sine = np.zeros((len(alphas), reservoir_size))
plotting_singular_values_periodic = np.zeros((len(alphas), reservoir_size))

# For each alpha value
for i, a in enumerate(alphas):
    # Get correlation matrices
    _, _, _, _, R1 = conceptor_collector.get(0)
    _, _, _, _, R2 = conceptor_collector.get(2)

    # Change aperture to a
    C1 = np.dot(R1, lin.inv(R1 + np.power(a, -2) * np.eye(reservoir_size)))
    U1, S1, V1 = lin.svd(C1)
    plotting_singular_values_sine[i] = S1

    # Change aperture to
    C2 = np.dot(R2, lin.inv(R2 + np.power(a, -2) * np.eye(reservoir_size)))
    U2, S2, V2 = lin.svd(C2)
    plotting_singular_values_periodic[i] = S2
# end for

# Compare singular values
Conceptors.visualization.compare_singular_values(
    singvalues1=plotting_singular_values_sine,
    singvalues2=plotting_singular_values_periodic,
    title1="Sine singular values",
    title2="Periodic singular values"
)
