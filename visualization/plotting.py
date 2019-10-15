# coding=utf-8
#
# File : plotting.py
# Description : Plot data.
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
import matplotlib.pyplot as plt
import numpy as np


# Plot to sets of singular values side by side.
def compare_singular_values(singvalues1, singvalues2, title1, title2):
    """
    Plot to sets of singular values side by side.
    :param singvalues1: First set of singular values.
    :param singvalues2: Second set of singular values.
    :param title1: First plot's title
    :param title2: Second plot's title
    """
    # Assert
    assert isinstance(singvalues1, np.ndarray)
    assert isinstance(singvalues2, np.ndarray)
    print(singvalues1.shape)
    assert singvalues1.ndim == 1
    assert singvalues2.ndim == 1

    # Figure (square size)
    plt.figure(figsize=(12, 4))

    # Plot sine singular values
    plt.subplot(1, 2, 1)
    for sv in singvalues1:
        plt.plot(sv, linewidth=2)
    # end for
    plt.title(title1)
    plt.yticks([0, 1])
    plt.ylim([0, 1.1])

    # Plot periodic
    plt.subplot(1, 2, 2)
    for sv in singvalues2:
        plt.plot(sv, linewidth=2)
    # end for
    plt.yticks([0, 1])
    plt.ylim([0, 1.1])
    plt.title(title2)

    # Show
    plt.show()
# end compare_singular_values


# Plot singular values
def plot_singular_values(singular_values, title):
    """
    Plot singular values
    :param singular_values: Singular values to plot
    :param title: Plot's title
    """
    # Assert
    assert isinstance(singular_values, np.ndarray)
    assert singular_values.ndim == 1

    # Figure (square size)
    plt.figure(figsize=(12, 4))

    # Plot sine singular values
    plt.subplot(1, 2, 1)
    for sv in singular_values:
        plt.plot(sv, linewidth=2)
    # end for
    plt.title(title)
    plt.yticks([0, 1])
    plt.ylim([0, 1.1])

    # Show
    plt.show()
# end plot_singular_values
