# coding=utf-8
#
# File : interpolation.py
# Description : Interpolation and alignment.
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
from scipy.interpolate import interp1d
import numpy.linalg as lin


# Interpolate and align two signals
def interpolation_alignment(truth_pattern, generated_pattern, interpolation_rate, dim=0, kind='quadratic'):
    """
    Interpolate and align two signals
    :param truth_pattern:
    :param generated_pattern:
    :param interpolation_rate:
    :param dim:
    :param kind: Interpolation type
    :return:
    """
    # Lengths
    truth_length = truth_pattern.shape[dim]
    generated_length = generated_pattern.shape[dim]

    # Quadratic interpolation functions
    truth_pattern_func = interp1d(np.arange(truth_pattern), truth_pattern, kind=kind)
    generated_pattern_func = interp1d(np.arange(generated_length), generated_pattern, kind=kind)

    # Get interpolated patterns
    truth_pattern_int = truth_pattern_func(np.arange(0, truth_length - 1.0, 1.0 / interpolation_rate))
    generated_pattern_int = generated_pattern_func(np.arange(0, generated_length - 1.0, 1.0 / interpolation_rate))

    # Generated interpolated pattern length
    L = generated_pattern_int.shape[0]

    # Truth interpolated pattern length
    M = truth_pattern_int.shape[0]

    # Save L2 distance for each phase shift
    phase_matches = np.zeros(L - M)

    # For each phase shift
    for phases_hift in range(L - M):
        phase_matches[phases_hift] = lin.norm(truth_pattern_int - generated_pattern_int[phases_hift:phases_hift + M])
    # end for

    # Best match
    max_ind = int(np.argmax(-phase_matches))

    # Get the generated output matching the original signal
    aligned_outputs = generated_pattern_int[
        np.arange(max_ind, max_ind + interpolation_rate * truth_length, interpolation_rate)
    ]

    # Get the position in the original signal
    coarse_max_ind = int(np.ceil(max_ind / interpolation_rate))

    # Returns
    return aligned_outputs, coarse_max_ind
# end interpolation_alignment
