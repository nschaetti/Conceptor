# -*- coding: utf-8 -*-
#
# File : incremental_loading_demo.py
# Description : Section 3.2 of the paper "Controlling RNN with Conceptors". Incremental loading of a reservoir.
# Date : 17th of January, 2019
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

# Imports
import numpy as np
import numpy.linalg as lin
import math
import argparse

# Conceptors
from Conceptors.patterns import patts
import Conceptors.reservoir
import Conceptors.measures
import Conceptors.conceptors
import Conceptors.tools
import Conceptors.visualization
import Conceptors.logic

# Operators
from Conceptors.logic import PHI

# Setting system params
seed = 1
learn_type = 2
reservoir_size = 100
spectral_radius = 1.5
input_scaling = 1.5
bias_scaling = 0.25

# Incremental loading learning (batch offline)
washout_length = 100
learn_length = 100
aperture = 1000

# Incremental loading learning (online adaptive)
adapt_length = 1500
adapt_rate = 0.02
error_plot_smooth_rate = 0.01

# Pattern readout learning
ridge_param_wout = 0.01

# C testing
conceptor_test_length = 200
conceptor_test_washout = 200

# How many singular values are plotted
n_plot_singular_values = 100

# Signal plot length
signal_plot_length = 20

# Setting patterns
patterns = [0, 1, 8, 10, 11, 0, 1, 8, 43, 38, 39, 12, 33, 15, 36, 35]

# Argument parsing
parser = argparse.ArgumentParser(prog="incremental_loading_demo", description=u"Memory management and incremental loading")
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

#
# Incremental learning
#

# Number of patterns
n_patterns = len(patterns)

# Save all patterns
pattern_collectors = np.zeros((n_patterns, learn_length))

# Pattern to plot
plotting_patterns = np.zeros((n_patterns, signal_plot_length))

# Quotas of A
quota_A = np.zeros(n_patterns)

# Quota of C
quota_C = np.zeros(n_patterns)

# Save C(all)
# A_collectors = np.zeros((n_patterns, reservoir_size, reservoir_size))
A_SVs_collectors = np.zeros((n_patterns, reservoir_size))

# Save states
state_collectors = np.zeros((n_patterns, reservoir_size, learn_length))

# Last states
last_states = np.zeros((reservoir_size, n_patterns))

# Collection of conceptors
conceptor_collector = Conceptors.conceptors.Collector(reservoir_size)

# Init D and Wout matrices
D = np.zeros((reservoir_size, reservoir_size))
Wout = np.zeros([1, reservoir_size])

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
    state_collector_old = Conceptors.reservoir.timeshift(state_collector, -1)

    # Save patterns
    pattern_collectors[p] = pattern_collector

    # Patterns to plot
    plotting_patterns[p] = pattern_collector[:signal_plot_length]

    # Train conceptor
    C, U, Snew, Sorg, R = Conceptors.conceptors.train(
        X=state_collector,
        aperture=aperture
    )

    # Save conceptor and singular values
    conceptor_collector.add(p, C, U, Snew, Sorg, R)

    # Save last state
    last_states[:, p] = x

    # Compute D increment
    D = Conceptors.reservoir.incremental_loading(
        D=D,
        X=state_collector_old,
        P=pattern_collector,
        Win=Win,
        A=conceptor_collector.A(),
        aperture=aperture,
        training_length=learn_length
    )

    # Compute Wout
    Wout = Conceptors.reservoir.incremental_training(
        Wout=Wout,
        X=state_collector,
        P=pattern_collector,
        A=conceptor_collector.A(),
        ridge_param=ridge_param_wout,
        training_length=learn_length
    )

    # Disjonction of all conceptor
    A = conceptor_collector.A()

    # Save A
    # A_collectors[p] = A
    AU, AS, AV = lin.svd(A)
    A_SVs_collectors[p] = AS

    # Quota of loaded reservoir
    quota_A[p] = Conceptors.conceptors.quota(A)

    # Quota of the resized conceptor
    quota_C[p] = Conceptors.conceptors.quota(PHI(C, aperture))
# end for

# Now we generate signal but with conceptors
# Save states and outputs
conceptor_test_states = np.zeros((n_patterns, reservoir_size, conceptor_test_length))
conceptor_test_output = np.zeros((n_patterns, conceptor_test_length))

# For each pattern
for p in range(n_patterns):
    # Get the corresponding conceptors
    C, _, _, _, _ = conceptor_collector.get(p)

    # Get the corresponding conceptors
    C = PHI(C, aperture)

    # x0
    x = np.random.randn(reservoir_size)

    # Free run with the loaded reservoir
    test_states, test_output = Conceptors.reservoir.free_run_input_simulation(
        x_start=x,
        W=Wstar,
        D=D,
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

# Plot pattern regenerated with quota and NRMSE
Conceptors.visualization.plot_patterns_with_infos(
    original_patterns=plotting_patterns,
    generated_patterns=conceptor_test_output_aligned,
    gauge=A_SVs_collectors,
    info1=NRMSE_aligned,
    info2=quota_A,
    title="p and y"
)
