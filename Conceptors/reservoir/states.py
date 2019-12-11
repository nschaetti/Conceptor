# coding=utf-8
#
# File : states.py
# Description : States utility functions.
# Date : 15th of October, 2019
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


# Compute timeshifted states
def timeshift(X, temp_dif):
    """
    Compute timeshifted states.
    :param X: States (Nx x Length)
    :param temp_dif: Time steps.
    :return: Timeshifted reservoir states.
    """
    # Dimensions
    reservoir_size = X.shape[0]
    length = X.shape[1]

    # Init.
    Xshifted = np.zeros((reservoir_size, length))

    # Time shifted states
    if temp_dif < 0:
        Xshifted[:, -temp_dif:] = X[:, :temp_dif]
    else:
        Xshifted[:, :-temp_dif] = X[:, temp_dif:]
    # end if

    return Xshifted
# end timeshift
