# coding=utf-8
#
# File : and_operator.py
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
import numpy.linalg as lin


# C AND B
def AND(C, B):
    """
    AND operator in conceptor logic
    :param C: First matrix (reservoir size x reservoir size)
    :param B: Second matrix (reservoir size x reservoir size)
    :return: C AND B (reservoir size x reservoir size)
    """
    # Assert type
    assert isinstance(C, np.ndarray)
    assert isinstance(B, np.ndarray)

    # Assert dimension
    assert C.ndim == 2
    assert B.ndim == 2

    # Dimension
    dim = C.shape[0]
    tol = 1e-14

    # SV on both conceptor
    (UC, SC, UtC) = lin.svd(C)
    (UB, SB, UtB) = lin.svd(B)

    # Get singular values
    dSC = SC
    dSB = SB

    # How many non-zero singular values
    numRankC = int(np.sum(1.0 * (dSC > tol)))
    numRankB = int(np.sum(1.0 * (dSB > tol)))

    # Select zero singular vector
    UC0 = UC[:, numRankC:]
    UB0 = UB[:, numRankB:]

    # SVD on UC0 + UB0
    # (W, Sigma, Wt) = lin.svd(np.dot(UC0, UC0.T) + np.dot(UB0, UB0.T))
    (W, Sigma, Wt) = lin.svd(UC0 @ UC0.T + UB0 @ UB0.T)

    # Number of non-zero SV
    numRankSigma = int(sum(1.0 * (Sigma > tol)))

    # Select zero singular vector
    Wgk = W[:, numRankSigma:]

    # C and B
    # Wgk * (Wgk^T * (C^-1 + B^-1 - I) * Wgk)^-1 * Wgk^T
    CandB = Wgk @ lin.inv(Wgk.T @ (lin.pinv(C, tol) + lin.pinv(B, tol) - np.eye(dim)) @ Wgk) @ Wgk.T

    # Assert outout
    assert isinstance(CandB, np.ndarray)
    assert CandB.ndim == 2
    assert CandB.shape == C.shape
    assert CandB.shape == B.shape

    return CandB
# end AND
