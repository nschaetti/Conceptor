# coding=utf-8
#
# File : run.py
# Description : Run reservoirs in normal mode with inputs.
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
import scipy.sparse


# Run an echo state network for some steps
def run(pattern, reservoir_size, x_start, Wstar, Win, Wbias, run_length, washout_length, dim=0):
    """
    Run an echo state network for some steps
    :param pattern: The pattern as a function to input into the ESN.
    :param x_start: The starting state x0 (Nx).
    :param Wstar: The internal weight matrix (Nx x Nx).
    :param Win: The input connection matrix (Ny x Nx).
    :param Wbias: The bias matrix (Nx).
    :param run_length: How many states to return (> 0).
    :param washout_length: How many states to ignore at the beginning ?
    :param dim: Position of the temporal dimension (default=0).
    :return: output states (Nx x run_length or run_length x Nx), input pattern (run_length)
    """
    # Assert types
    assert callable(pattern)
    assert isinstance(reservoir_size, int)
    assert isinstance(x_start, np.ndarray)
    assert isinstance(Wstar, np.ndarray) or isinstance(Wstar, scipy.sparse.csc_matrix)
    assert isinstance(Win, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(run_length, int)
    assert isinstance(washout_length, int)
    assert isinstance(dim, int)

    # Dimension and values
    assert x_start.ndim == 1
    assert Wstar.ndim == 2
    assert Win.ndim == 2
    assert Wbias.ndim == 1
    assert run_length > 0
    assert washout_length >= 0
    assert dim == 0 or dim == 1

    # Squared matrices
    assert Wstar.shape[0] == Wstar.shape[1]

    # State collector
    if dim == 0:
        state_collector = np.zeros((reservoir_size, run_length))
    else:
        state_collector = np.zeros((run_length, reservoir_size))
    # end if

    # Pattern collector
    pattern_collector = np.zeros(run_length)

    # Starting state x0
    x = x_start

    # For each timestep
    for t in range(washout_length + run_length):
        # Get input u_t
        u = np.array([pattern(t)])

        # Compute ESN equation
        x = np.tanh(Wstar.dot(x.T) + Win.dot(u) + Wbias)
        x = np.asarray(x).flatten()

        # If washout ended, save
        if t >= washout_length:
            if dim == 0:
                state_collector[:, t - washout_length] = x
            else:
                state_collector[t - washout_length, :] = x
            # end if
            pattern_collector[t - washout_length] = u
        # end if
    # end if

    return state_collector, pattern_collector
# end run
