# coding=utf-8
#
# File : generate.py
# Description : Generate pattern from loaded reservoir and conceptors.
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


# Free run, run reservoir without inputs
def free_run(X_start, W, D, Wbias, Wout, C, run_length, washout_length):
    """
    Free run, run reservoir without inputs
    :param X_start: Starting state
    :param W:
    :param D:
    :param Wbias:
    :param Wout:
    :param C:
    :param run_length:
    :param washout_length:
    :return:
    """
    # Test states and outputs
    run_states = np.zeros((run_length, W.shape[0]))
    run_outputs = np.zeros(run_length)

    # Reservoir initial state
    x = X_start

    # For all timesteps
    for t in range(washout_length + run_length):
        # Compute tanh(W * x + bias)
        x = np.tanh(np.dot(W, x) + np.dot(D, x) + Wbias)
        x = np.asarray(x).flatten()

        # Filter through conceptor (C * x)
        x = np.dot(C, x)

        # Save states and outputs if outside washout
        if t >= washout_length:
            run_states[t - washout_length, :] = x
            run_outputs[t - washout_length] = np.dot(Wout, x)
        # end if
    # end for

    return run_states, run_outputs
# end free_run
