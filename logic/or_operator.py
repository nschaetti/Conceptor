# coding=utf-8
#
# File : or_operator.py
# Description : OR in Conceptor Logic
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
from .and_operator import AND
from .not_operator import NOT


# OR in Conceptor Logic
def OR(R, Q):
    """
    OR in Conceptor Logic
    :param R: First conceptor operand (reservoir size x reservoir size)
    :param Q: Second conceptor operand (reservoir size x reservoir size)
    :return: R OR Q
    """
    # Assert type
    assert isinstance(R, np.ndarray)
    assert isinstance(Q, np.ndarray)
    assert R.ndim == 2
    assert Q.ndim == 2

    # R OR Q
    RorQ = NOT(AND(NOT(R), NOT(Q)))

    return RorQ
# end OR
