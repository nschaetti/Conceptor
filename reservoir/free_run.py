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


# Run reservoir with internal loaded matrix W
def free_run(x_start, W, Wbias, Wout, C, run_length, washout_length, dim=0):
    """
    Run reservoir with internal loaded matrix W
    :param x_start: Start state (Nx)
    :param W: Internal weight matrix (Nx x Nx)
    :param Wbias: Bias matrix (Nx size)
    :param Wout: Trained output matrix (Ny x Nx)
    :param C: Conceptor (can be none) (Nx x Nx)
    :param run_length: How many timesteps to generate (> 0)?
    :param washout_length: How many timesteps to ignore at the beginning?
    :param dim: Position of the temporal dimension (default=0).
    :return: resulting states (run_length x Nx OR Nx x run_length, generated outputs (run_length)
    """
    # Assert types
    assert isinstance(x_start, np.ndarray)
    assert isinstance(W, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(Wout, np.ndarray)
    assert isinstance(C, np.ndarray) or C is None
    assert isinstance(run_length, int)
    assert isinstance(washout_length, int)
    assert isinstance(dim, int)

    # Dimension and values
    assert x_start.ndim == 1
    assert W.ndim == 2
    assert Wbias.ndim == 1
    assert Wout.ndim == 2
    assert C is None or C.ndim == 2
    assert run_length > 0
    assert washout_length >= 0
    assert dim == 0 or dim == 1

    # Squared matrices
    assert W.shape[0] == W.shape[1]
    assert C is None or C.shape[0] == C.shape[1]

    # Test states and outputs
    if dim == 0:
        run_states = np.zeros((run_length, W.shape[0]))
        run_outputs = np.zeros(run_length)
    else:
        run_states = np.zeros((W.shape[0], run_length))
        run_outputs = np.zeros(run_length)
    # end if

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
        if t >= washout_length:
            if dim == 0:
                run_states[t - washout_length, :] = x
            else:
                run_states[:, t - washout_length] = x
            # end if
            run_outputs[t - washout_length] = Wout @ x
        # end if
    # end for

    return run_states, run_outputs
# end free_run


# Run reservoir with input simulation matrix D
def free_run_input_simulation(x_start, W, D, Wbias, Wout, C, run_length, washout_length, dim=0):
    """
    Run reservoir without input simulation matrix D
    :param x_start: Starting state (Nx)
    :param W: Internal weight matrix (Nx x Nx)
    :param D: Input simulation matrix (Nx x Nx)
    :param Wbias: Bias matrix (Nx)
    :param Wout: Trained output matrix (Ny x Nx)
    :param C: Conceptor (can be None) (Nx x Nx)
    :param run_length: How many timesteps to run (> 0).
    :param washout_length: How many timesteps to ignore at the beginning?
    :param dim: Position of the temporal dimension (default=0).
    :return: resulting states  (run_length x Nx OR run_length x run_length), generated outputs (run_length)
    """
    # Assert types
    assert isinstance(x_start, np.ndarray)
    assert isinstance(W, np.ndarray)
    assert isinstance(D, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(Wout, np.ndarray)
    assert isinstance(C, np.ndarray) or C is None
    assert isinstance(run_length, int)
    assert isinstance(washout_length, int)
    assert isinstance(dim, int)

    # Dimension and values
    assert x_start.ndim == 1
    assert W.ndim == 2
    assert D.ndim == 2
    assert Wbias.ndim == 1
    assert Wout.ndim == 2
    assert C is None or C.ndim == 2
    assert run_length > 0
    assert washout_length >= 0
    assert dim == 0 or dim == 1

    # Squared matrices
    assert W.shape[0] == W.shape[1]
    assert D.shape[0] == D.shape[1]
    assert C is None or C.shape[0] == C.shape[1]

    # Test states and outputs
    if dim == 0:
        run_states = np.zeros((run_length, W.shape[0]))
        run_outputs = np.zeros(run_length)
    else:
        run_states = np.zeros((W.shape[0], run_length))
        run_outputs = np.zeros(run_length)
    # end if

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
            if dim == 0:
                run_states[t - washout_length, :] = x
            else:
                run_states[:, t - washout_length] = x
            # end if
            run_outputs[t - washout_length] = Wout @ x
        # end if
    # end for

    return run_states, run_outputs
# end free_run_input_simulation


# Run reservoir with input recreation matrix R
def free_run_input_recreation(x_start, W, R, Win, Wbias, Wout, C, run_length, washout_length, dim=0):
    """
    Run reservoir with input recreation matrix R
    :param x_start: Starting state (reservoir size)
    :param W: Internal weight matrix (reservoir size x reservoir size)
    :param R: Input recreation matrix (# inputs x reservoir size)
    :param Win: Input weight matrix (reservoir size x # inputs)
    :param Wbias: Bias matrix (reservoir size)
    :param Wout: Trained output matrix (reservoir size x # outputs)
    :param C: Conceptor (can be none) (reservoir size x reservoir size)
    :param run_length: How many timestep to run
    :param washout_length: How many timesteps to ignore at the beginning.
    :param dim: Position of the temporal dimension (default=0).
    :return: resulting state, generated outputs
    """
    # Assert types
    assert isinstance(x_start, np.ndarray)
    assert isinstance(W, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(Win, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(Wout, np.ndarray)
    assert isinstance(C, np.ndarray) or C is None
    assert isinstance(run_length, int)
    assert isinstance(washout_length, int)
    assert isinstance(dim, int)

    # Dimension and values
    assert x_start.ndim == 1
    assert W.ndim == 2
    assert R.ndim == 2
    assert Win.ndim == 2
    assert Wbias.ndim == 1
    assert Wout.ndim == 2
    assert C is None or C.ndim == 2
    assert run_length > 0
    assert washout_length >= 0
    assert dim == 0 or dim == 1

    # Squared matrices
    assert W.shape[0] == W.shape[1]
    assert C is None or C.shape[0] == C.shape[1]

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
