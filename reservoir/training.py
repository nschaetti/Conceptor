# coding=utf-8
#
# File : generate.py
# Description : Generate pattern from loaded reservoir and conceptors.
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


# Ridge regression
def ridge_regression(X, Y, ridge_param, dim=0):
    """
    Ridge regression
    :param X: Samples
    :param Y: Observations
    :param ridge_param: Regularization parameters
    :param dim: Dimension index of the samples
    :return: Least square solution to Xb + l = Y
    """
    # Size
    if dim == 0:
        n_predictors = X.shape[1]
    else:
        n_predictors = X.shape[0]
    # end if

    # Compute solution
    if dim == 0:
        B = lin.inv(X.T @ X + ridge_param * np.eye(n_predictors)) @ X @ Y
    else:
        B = (lin.inv(X @ X.T + ridge_param * np.eye(n_predictors)) @ X @ Y.T).T
    # end if

    return B
# end ridge_regression


# Train outputs
def train_outputs(training_states, training_targets, ridge_param_wout, dim=0):
    """
    Train ESN outputs
    :param training_states: Training states
    :param training_targets: Training targets
    :param ridge_param_wout: Regularization parameters
    :param dim: Which dimension corresponds to time.
    :return: Trained Wout matrix
    """
    # Solve Ax = b problem
    return ridge_regression(training_states, training_targets, ridge_param_wout, dim)
# end train_outputs


