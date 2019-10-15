# coding=utf-8
#
# File : pattern_visualization.py
# Description : View and analyse patterns.
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
import matplotlib.pyplot as plt


# Plot original and generated patterns with singular values and reservoir states
def plot_patterns_with_singular_values(truth_patterns, generated_patterns, Xs, SVs, color_truth='b', color_generated='r', title='p and y'):
    """
    Plot original and generated patterns with singular values and reservoir states.
    :param truth_patterns: Original patterns (#patterns x L)
    :param generated_patterns: Generated patterns (#patterns x L)
    :param Xs: Reservoir states limited to 3 neurons (#patterns x 3)
    :param SVs: Singular values (Nx)
    :param color_truth: Color line of the original signal
    :param color_generated: Color line of the generated signal
    :param title: Plot title
    """
    # Assert types
    assert isinstance(truth_patterns, np.ndarray)
    assert isinstance(generated_patterns, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(SVs, np.ndarray)
    assert isinstance(color_truth, str)
    assert isinstance(color_generated, str)
    assert isinstance(title, str)

    # Assert dimension
    assert truth_patterns.ndim == 2
    assert generated_patterns.ndim == 2
    assert Xs.ndim == 3

    # N. patterns
    n_patterns = truth_patterns.shape[0]

    # Assert
    assert n_patterns > 0

    # Figure (square size)
    plt.figure(figsize=(16, 2 * n_patterns))

    # For each pattern
    for p in range(n_patterns):
        # Plot 1 : original pattern and recreated pattern
        plt.subplot(n_patterns, 4, p * 4 + 1)
        plt.plot(generated_patterns[p], color=color_generated, linewidth=5)
        plt.plot(truth_patterns[p], color=color_truth, linewidth=1.5)

        # Title
        if p == 0:
            plt.title(title)
        # end if

        # X labels
        if p == 3:
            plt.xticks([0, 10, 20])
        else:
            plt.xticks([])
        # end if

        # Y limits
        plt.ylim([-1, 1])
        plt.yticks([-1, 0, 1])

        # Plot 2 : states
        plt.subplot(n_patterns, 4, p * 4 + 2)
        plt.plot(Xs[p, 0])
        plt.plot(Xs[p, 1])
        plt.plot(Xs[p, 2])

        # Title
        if p == 0:
            plt.title(u'two neurons')
        # end if

        # X labels
        if p == 3:
            plt.xticks([0, 10, 20])
        else:
            plt.xticks([])
        # end if

        # Y limits
        plt.ylim([-1, 1])
        plt.yticks([-1, 0, 1])

        # Plot 3 : Log10 of singular values (PC energy)
        plt.subplot(n_patterns, 4, p * 4 + 3)
        plt.plot(np.log10(SVs[p]), 'red', linewidth=2)

        # X labels
        if p == 3:
            plt.xticks([0, 50, 100])
        else:
            plt.xticks([])
        # end if

        # Limits
        plt.ylim([-20, 10])

        # Title
        if p == 0:
            plt.title(u'log10 PC Energy')
        # end if

        # Learning PC energy
        plt.subplot(n_patterns, 4, p * 4 + 4)
        plt.plot(SVs[p, :10], 'red', linewidth=2)

        # Title
        if p == 0:
            plt.title(u"leading PC energy")
        # end ifS

        # Limits
        plt.ylim([0, 40.0])
    # end for

    # Show figure
    plt.show()
# end plot_patterns_with_singular_values


# Plot original and generated patterns on the same plot, with informations
def plot_patterns_with_infos(original_patterns, generated_patterns, gauge, info1, info2, title):
    """
    Plot original and generated patterns on the same plot, with informations
    :param original_patterns: Original patterns (#patterns x length)
    :param generated_patterns: Generated patterns (#patterns x length)
    :param gauge: A gauge to dislay at the back of the plots (length)
    :param info1: First info to display, numbers (#patterns)
    :param info2: Second info to display, numbers (#patterns)
    :param title: Plot's title.
    """
    # how many patterns
    n_patterns = original_patterns.shape[0]
    pattern_length = original_patterns.shape[1]

    # Figure (square size)
    plt.figure(figsize=(16, 4 * n_patterns))

    # Plot index
    plot_index = 0

    # For each pattern
    for p in range(n_patterns):
        # Plot 1 : original pattern and recreated pattern
        plt.subplot(4, 4, plot_index + 1)
        plot_index += 1

        # Gauge length
        gauge_length = gauge.shape[1]

        # Plot singular values of A
        plt.fill_between(
            # np.linspace(0, pattern_length, n_plot_singular_values),
            # 2.0 * Sx - 1.0,
            np.linspace(0, pattern_length, gauge_length),
            2.0 * gauge[p] - 1.0,
            -1,
            color='red',
            alpha=0.75
        )

        # Plot generated pattern and original
        plt.plot(generated_patterns[p], color='lime', linewidth=10)
        plt.plot(original_patterns[p], color='black', linewidth=1.5)

        # Square properties
        plot_width = pattern_length
        plot_bottom = -1
        plot_top = 1
        props = dict(boxstyle='square', facecolor='white', alpha=0.75)

        # Pattern number
        plt.text(
            plot_width - 0.7,
            plot_top - 0.1,
            "p = {}".format(p+1),
            fontsize=14,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props
        )

        # Info 1
        plt.text(
            plot_width - 0.7,
            plot_bottom + 0.1,
            round(info1[p], 4),
            fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=props
        )

        # Info 2
        plt.text(
            0.7,
            plot_bottom + 0.1,
            round(info2[p], 2),
            fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=props
        )

        # Title
        if p == 0:
            plt.title(title)
        # end if

        # X labels
        if p == 3:
            plt.xticks([0, pattern_length / 2.0, pattern_length])
        else:
            plt.xticks([0, pattern_length / 2.0, pattern_length])
        # end if

        # Y limits
        plt.ylim([-1, 1])
        plt.xlim([0, pattern_length])
        plt.yticks([-1, 0, 1])
    # end for

    # Show figure
    plt.show()
# end plot_patterns_with_infos

