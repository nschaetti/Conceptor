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
import numpy.linalg as lin


# C AND B
def AND(C, B):
    """
    C AND B
    :param C:
    :param B:
    :return:
    """
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
    (W, Sigma, Wt) = lin.svd(np.dot(UC0, UC0.T) + np.dot(UB0, UB0.T))

    # Number of non-zero SV
    numRankSigma = int(sum(1.0 * (Sigma > tol)))

    # Select zero singular vector
    Wgk = W[:, numRankSigma:]

    # C and B
    # Wgk * (Wgk^T * (C^-1 + B^-1 - I) * Wgk)^-1 * Wgk^T
    CandB = np.dot(
        np.dot(
            Wgk,
            lin.inv(
                np.dot(
                    np.dot(
                        Wgk.T,
                        (lin.pinv(C, tol) + lin.pinv(B, tol) - np.eye(dim))
                    ),
                    Wgk
                )
            )
        ),
        Wgk.T
    )

    return CandB
# end AND
