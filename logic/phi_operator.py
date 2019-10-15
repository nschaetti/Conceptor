# coding=utf-8
#
# File : phi_operator.py
# Description : Aperture multiplication in Conceptor Logic
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
import math
from scipy.linalg import svd


# Aperture adaptation of conceptor C by factor gamma where 0 <= gamma <= Inf.
def PHI(C, gamma):
    """
    Aperture adaptation of conceptor C by factor gamma where 0 <= gamma <= Inf.
    :param C: Conceptor matrix (reservoir size x reservoir size)
    :param gamma: Multiplication factor (0 <= gamma <= Inf)
    :return: C with aperture multiplied by gamma
    """
    # Assert type
    assert isinstance(C, np.ndarray)
    assert isinstance(gamma, float) or isinstance(gamma, int)

    # Dim and value
    assert C.ndim == 2
    assert gamma >= 0 and gamma <= float("inf")

    # Dimension
    dim = C.shape[0]

    # Multiply by 0
    if gamma == 0:
        (U, S, V) = svd(C)
        Sdiag = S
        Sdiag[Sdiag < 1] = np.zeros((sum(Sdiag < 1), 1))
        Cnew = U @ np.diag(Sdiag) @ U.T
    elif gamma == float("inf"):
        (U, S, V) = svd(C)
        Sdiag = S
        Sdiag[Sdiag > 0] = np.ones(sum(Sdiag > 0), 1)
        Cnew = U @ np.diag(Sdiag) @ U.T
    else:
        Cnew = C @ lin.inv(C + math.pow(gamma, -2) * (np.eye(dim) - C))
    # end

    return Cnew
# end PHI
