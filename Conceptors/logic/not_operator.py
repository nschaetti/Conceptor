# coding=utf-8
#
# File : not_operator.py
# Description : NOT in Conceptor Logic
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


# NOT operator in Conceptor Logic
def NOT(R):
    """
    NOT operator in Conceptor Logic
    :param R: Matrix operand (reservoir size x reservoir size
    :return:
    """
    # Assert R
    assert isinstance(R, np.ndarray)
    assert R.ndim == 2

    # NOT => I - R
    dim = R.shape[0]
    assert dim > 0
    notR = np.eye(dim) - R

    return notR
# end NOT
