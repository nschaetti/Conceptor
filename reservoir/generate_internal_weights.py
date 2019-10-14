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
from numpy import linalg as lin
import scipy.sparse


# Normally distributed matrix values
def random_matrix_values(n):
    return np.random.randn(n)
# end random_matrix_values


# Generate internal weights
def generate_internal_weights(n_internal_units, connectivity, seed=1):
    """
    Create a random sparse reservoir for an ESN. Nonzero weights are normal distributed.
    :param n_internal_units: the number of internal units in the ESN
    :param connectivity: a real in [0, 1], the (rough) proportion of nonzero weights
    :return: matrix of size n_internal_units * n_internal_units
    """
    # Generate sparse matrix
    internal_weights = scipy.sparse.random(n_internal_units, n_internal_units, connectivity, random_state=seed, data_rvs=random_matrix_values).todense()
    spectral_radius = abs(lin.eig(internal_weights)[0])[0]
    return internal_weights / spectral_radius
# end generate_internal_weights
