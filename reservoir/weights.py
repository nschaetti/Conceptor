# coding=utf-8
#
# File : weights.py
# Description : Manage weights and hyperparameters
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
import scipy.io as io


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


# Change rate according to hyper-parameters
def scale_weights(W, Win, Wbias, spectral_radius, input_scaling, bias_scaling):
    """
    Change rate according to hyper-parameters
    :param W: Internal weights
    :param Win: Input weights
    :param Wbias: Bias
    :param spectral_radius: Spectral radius
    :param input_scaling: Input scaling
    :param bias_scaling: Bias scaling
    :return: W, Win, Wbias
    """
    return W * spectral_radius, Win * input_scaling, Wbias * bias_scaling
# end scale_weights


# Load matrix from matlab file
def load_matlab_file(file_name, entity_name):
    """
    Load matrix matlab file
    :param file_name: Matlab file
    :param entity_name: Entry name to load.
    :return: Loaded matrix
    """
    return io.loadmat(file_name)[entity_name]
# end load_matlab_file


# Load weights from matlab
def from_matlab(w_file, w_name, win_file, win_name,  wbias_file, wbias_name):
    """
    Load weights from matlab
    :param w_file: Internal weights file
    :param w_name: Name of the internal weights matrix in the file
    :param win_file:
    :param win_name:
    :param wbias_file:
    :param wbias_name:
    :return:
    """
    # Load internal weights
    W_raw = load_matlab_file(w_file, w_name).todense()

    # Load Win and Wbias
    Win_raw = load_matlab_file(win_file, win_name).reshape(-1, 1)
    Wbias_raw = load_matlab_file(wbias_file, wbias_name).reshape(-1)

    return W_raw, Win_raw, Wbias_raw
# end from_matlab
