# coding=utf-8
#
# File : collector.py
# Description : A tool class to stock and manage Conceptors
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
from Conceptors.logic import OR


# Collector class
class Collector:
    """
    Collector class
    """

    # Constructor
    def __init__(self, reservoir_size):
        """
        Constructor
        :param reservoir_size: Reservoir size (number of units)
        """
        # Assert type and value
        assert isinstance(reservoir_size, int)
        assert reservoir_size > 0

        # Parameters
        self.reservoir_size = reservoir_size

        # Empty list
        self.conceptors = dict()

        # Init. A
        self._A = np.zeros((reservoir_size, reservoir_size))
    # end __init__

    ###########################
    # PUBLIC
    ###########################

    # Add a conceptor and USU.T
    def add(self, index, C, U, Snorm, Sorg, R):
        """
        Append a conceptor and its description.
        :param index: Index as integer refering to the conceptor inserted.
        :param C: Conceptor matrix (reservoir size x reservoir size).
        :param U: Singular vectors (reservoir size x reservoir size).
        :param Snorm: Normalized singular values (reservoir size)
        :param Sorg: Original singular values (reservoir size)
        :param R: Correlation matrix (reservoir size x reservoir size)
        """
        # Assert type
        assert isinstance(index, int)
        assert isinstance(C, np.ndarray)
        assert isinstance(U, np.ndarray)
        assert isinstance(Snorm, np.ndarray)
        assert isinstance(Sorg, np.ndarray)
        assert isinstance(R, np.ndarray)

        # Assert dims
        assert C.ndim == 2
        assert U.ndim == 2
        assert Snorm.ndim == 1
        assert Sorg.ndim == 1
        assert R.ndim == 2

        # Assert size
        assert C.shape == (self.reservoir_size, self.reservoir_size)
        assert U.shape == (self.reservoir_size, self.reservoir_size)
        assert Snorm.shape[0] == self.reservoir_size
        assert Sorg.shape[0] == self.reservoir_size
        assert R.shape == (self.reservoir_size, self.reservoir_size)

        # Add to list
        self.conceptors[index] = (C, U, Snorm, Sorg, R)

        # Append to A
        self._A = OR(self._A, C)
    # end add

    # Get conceptor
    def get(self, index):
        """
        Get conceptor from index
        :param index:
        :return: (C, U, S normalized, S original, R)
        """
        return self.conceptors[index]
    # end get

    # Reset conceptor list and A
    def reset(self):
        """
        Reset conceptor list and A
        """
        # Reset A
        self._A = np.zeros((self.reservoir_size, self.reservoir_size))

        # Reset list
        self.conceptors = dict()
    # end reset

    # Get all conceptor matrix A
    def A(self):
        """
        Get all conceptor matrix A
        :return: Matrix A (reservoir size x reservoir size)
        """
        return self._A
    # end A

# end Collector
