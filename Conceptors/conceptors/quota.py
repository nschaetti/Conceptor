# coding=utf-8
#
# File : quota.py
# Description : Quota utility functions
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
import numpy.linalg as lin


# Compute quota of a matrix
def quota(M):
    """
    Compute quota of a matrix
    :param M: Square matrix (Nx x Nx)
    :return: Matrix's quotas 0 <= quota <= 1
    """
    # Assert
    assert isinstance(M, np.ndarray)
    assert M.ndim == 2
    assert M.shape[0] == M.shape[1]

    # SVD the matrix
    U, S, V = lin.svd(M)

    # Average of singular values
    return np.average(S)
# end quota
