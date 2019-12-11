# coding=utf-8
#
# File : training.py
# Description : Functions to train Conceptors
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
from numpy.linalg import svd
import math
import numpy.linalg as lin


# Train conceptor
def train(X, aperture):
    """
    Train a conceptor
    :param X: Reservoir states (L x Nx)
    :param aperture: Aperture (>= 0)
    :return: Trained conceptor matrix C, Normalized singular values Snorm, Singular values Sorg, Correlation matrix R
    """
    # Assert type
    assert isinstance(X, np.ndarray)
    assert isinstance(aperture, float) or isinstance(aperture, int)

    # Assert dimension and value
    assert X.ndim == 2
    assert aperture >= 0

    # Learn length
    learn_length = X.shape[1]
    reservoir_size = X.shape[0]

    # Assert learn length and
    # reservoir size.
    assert learn_length > 0
    assert reservoir_size > 0

    # CoRrelation matrix of reservoir states
    R = (X @ X.T) / float(learn_length)

    # Compute SVD on R
    Ux, Sx, Vx = svd(R)

    # Compute new singular values
    Snew = np.diag(Sx) @ lin.inv(np.diag(Sx) + math.pow(aperture, -2) * np.eye(reservoir_size))

    # Apply new SVs to get the conceptor
    return Ux @ Snew @ Ux.T, Ux, np.diag(Snew), Sx, R
# end train
