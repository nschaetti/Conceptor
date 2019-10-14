# coding=utf-8
#
# File : free_run.py
# Description : Run reservoirs in free-run mode with loaded internal weights and conceptors.
# Date : 14th of October, 2019
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


# Free run, run reservoir with internal loaded matrix W
def free_run(x_start, W, Wbias, Wout, C, run_length, washout_length):
    """
    Free run, run reservoir with internal loaded matrix W
    :param x_start: Start state
    :param W: Internal weight matrix
    :param Wbias: Bias matrix
    :param Wout: Trained output matrix
    :param C: Conceptor (can be none)
    :param run_length: How many timesteps to generate?
    :param washout_length: How many timesteps to ignore at the beginning?
    :return: resulting states, generated outputs
    """
    # Test states and outputs
    run_states = np.zeros((run_length, W.shape[0]))
    run_outputs = np.zeros(run_length)

    # Initial state x0
    x = x_start

    # For all timesteps
    for t in range(run_length):
        # Compute tanh(W * x + bias)
        x = np.tanh(W @ x + Wbias)

        # Filter through conceptor (C * x)
        if C is not None:
            x = C @ x
        # end if

        # Save states and outputs
        run_states[t, :] = x
        run_outputs[t] = Wout @ x
    # end for

    return run_states, run_outputs
# end free_run


# Free run, run reservoir with input simulation matrix D
def free_run_input_simulation(x_start, W, D, Wbias, Wout, C, run_length, washout_length):
    """
    Free run, run reservoir without inputs
    :param x_start: Starting state
    :param W: Internal weight matrix
    :param D: Input simulation matrix
    :param Wbias: Bias matrix
    :param Wout: Trained output matrix
    :param C: Conceptor (can be None)
    :param run_length: How many timesteps to run.
    :param washout_length: How many timesteps to ignore.
    :return: resulting states, generated outputs
    """
    # Test states and outputs
    run_states = np.zeros((run_length, W.shape[0]))
    run_outputs = np.zeros(run_length)

    # Reservoir initial state
    x = x_start

    # For all timesteps
    for t in range(washout_length + run_length):
        # Compute tanh(W * x + D * x + bias)
        x = np.tanh(W @ x + D @ x + Wbias)
        x = np.asarray(x).flatten()

        # Filter through conceptor (C * x)
        if C is not None:
            x = C @ x
        # end if

        # Save states and outputs if outside washout
        if t >= washout_length:
            run_states[t - washout_length, :] = x
            run_outputs[t - washout_length] = Wout @ x
        # end if
    # end for

    return run_states, run_outputs
# end free_run_input_simulation


# Free run, run reservoir with input recreation matrix R
def free_run_input_recreation(x_start, W, R, Win, Wbias, Wout, C, run_length, washout_length):
    """
    Run reservoir with input recreation matrix R
    :param x_start: Starting state
    :param W: Internal weight matrix
    :param R: Input recreation matrix
    :param Win: Input weight matrix
    :param Wbias: Bias matrix
    :param Wout: Trained output matrix
    :param C: Conceptor (can be none)
    :param run_length: How many timestep to run
    :param washout_length: How many timesteps to ignore at the beginning.
    :return: resulting state, generated outputs
    """
    # Test states and outputs
    run_states = np.zeros((run_length, W.shape[0]))
    run_outputs = np.zeros(run_length)

    # Reservoir initial state
    x = x_start

    # For all timesteps
    for t in range(washout_length + run_length):
        # Compute tanh(W * x + Win * R * x + bias)
        x = np.tanh(W @ x + Win @ (R @ x) + Wbias)
        x = np.asarray(x).flatten()

        # Filter through conceptor (C * x)
        if C is not None:
            x = C @ x
        # end if

        # Save states and outputs if outside washout
        if t >= washout_length:
            run_states[t - washout_length, :] = x
            run_outputs[t - washout_length] = np.dot(Wout, x)
        # end if
    # end for

    return run_states, run_outputs
# end free_run_input_recreation
