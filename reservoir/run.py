# coding=utf-8
#
# File : and.py
# Description : AND in Conceptor Logic
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


# Run an echo state network for some steps
def run(pattern, reservoir_size, x_start, Wstar, Win, Wbias, training_length, washout_length):
    """
    Run an echo state network for some steps
    :param pattern: The pattern as a function to input into the ESN.
    :param x_start: The starting state x0.
    :param Wstar: The internal weight matrix.
    :param Win: The input connection matrix.
    :param Wbias: The bias matrix.
    :param training_length: How many states to return.
    :param washout_length: How many states to ignore.
    :return: output states, input pattern
    """
    # State collector
    state_collector = np.zeros((reservoir_size, training_length))

    # Pattern collector
    pattern_collector = np.zeros(training_length)

    # Starting state x0
    x = x_start

    # For each timestep
    for t in range(washout_length + training_length):
        # Get input u_t
        u = np.array([pattern(t)])

        # Save x for old states
        x_old = x

        # Compute ESN equation
        x = np.tanh(Wstar.dot(x.T) + Win.dot(u) + Wbias)
        x = np.asarray(x).flatten()

        # If washout ended, save
        if t >= washout_length:
            state_collector[:, t - washout_length] = x
            pattern_collector[t - washout_length] = u
        # end if
    # end if

    return state_collector, pattern_collector
# end run
