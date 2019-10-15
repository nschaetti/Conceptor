# coding=utf-8
#
# File : nrmse.py
# Description : Normalized Root Mean Squared error
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


# Normalized Root Mean Square Error
def nrmse(predicted, target, norm="sd", axis=1):
    """
    Compute normalized root mean square error. If timeseries if more than 1-D, NRMSE is averaged over dimensions.
    :param predicted: Predicted timeseries (D x L)
    :param target: True timeseries (D x L)
    :param norm: Which norm to use (sd or maxmin)
    :return: Normalized Root Mean Square Error
    """
    # Assert types
    assert isinstance(predicted, np.ndarray)
    assert isinstance(target, np.ndarray)

    # Assert dim
    assert predicted.ndim == 2
    assert target.ndim == 2

    # Norm
    if norm == "maxmin":
        return np.average(np.divide(
            np.sqrt(np.average(np.power(predicted - target, 2), axis=axis)),
            np.max(target, axis=axis) - np.min(target, axis=axis)
        ))
    else:
        combined_var = 0.5 * (np.var(target, ddof=1, axis=axis) + np.var(predicted, ddof=1, axis=axis))
        return np.average(np.sqrt(np.divide(np.average(np.power(predicted - target, 2), axis=axis), combined_var)))
    # end if
# end nrmse


# NRMSE aligned
def nrmse_aligned(reference, signal, variance, interpolation_rate):
    """
    NRMSE aligned
    :param reference:
    :param signal:
    :param variance:
    :param interpolation_rate:
    :return:
    """
    # Lengths
    signal_length = len(signal)
    reference_length = len(reference)

    # Interpolation
    signal_interpolation = interp1d(np.arange(signal_length), signal, kind='quadratic')
    reference_interpolation = interp1d(np.arange(reference_length), reference, kind='quadratic')

    # Length
    L = reference_interpolation.shape[1]
    M = signal_interpolation.shape[1]

    # Phase matches
    phase_matches = np.zeros((1, L - M))

    # For each phase shift
    for phaseshift in range(L - M):
        phase_matches[0, phaseshift] = np.linalg.norm(signal_interpolation - reference_interpolation[0, np.arange(phaseshift, phaseshift + M - 1)])
    # end for

    # Maximum
    max_val, max_index = np.max(-phase_matches)

    # Reference aligned
    reference_aligned = reference_interpolation[0, np.arange(max_index, max_index + interpolation_rate * signal_length - 1, interpolation_rate)]

    return np.sqrt(np.average(np.power(signal - reference_aligned, 2) / variance))
# end nrmse_aligned

