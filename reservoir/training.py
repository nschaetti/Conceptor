# coding=utf-8
#
# File : training.py
# Description : Training methods and linear solvers.
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
from logic import NOT
from numpy.linalg import inv, pinv


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


# Reservoir loading
def loading(X, Xold, Wbias, ridge_param, dim=0):
    """
    Reservoir loading
    :param X: Reservoir states
    :param Xold: Timeshifted stated (-1)
    :param Wbias: Bias matrix
    :param ridge_param: Ridge parameter
    :param dim: Time dimension
    :return: Loaded internal weight matrix, learned states
    """
    # Reservoir size
    if dim == 0:
        learn_length = X.shape[0]
        reservoir_size = X.shape[1]
    else:
        learn_length = X.shape[1]
        reservoir_size = X.shape[0]
    # end if

    # Target for learning W
    X_new = (
            np.arctanh(X) - np.repeat(Wbias.reshape(1, -1), learn_length, axis=0).T
    )

    # Learning W
    return (
        (lin.inv(Xold @ Xold.T + ridge_param * np.eye(reservoir_size)) @ Xold) @ X_new.T
    ).T, X_new
# end loading


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


# Incremental loading
def incremental_loading(D, X, P, Win, A, aperture, training_length):
    """
    Incremental lodaing
    :param D: Current input simulation matrix
    :param X: Training states
    :param P: Original pattern
    :param Win: Input weight matrix
    :param A: Current disjonction of all conceptors loaded
    :param training_length: Training length
    :return: The new input simulation matrix D
    """
    # Compute D increment
    Td = np.dot(Win, P.reshape(1, -1)) - np.dot(D, X)

    # The linear subspace of the reservoir state space that are not yet
    # occupied by any pattern.
    F = NOT(A)

    # Compute the increment for matrix D
    S_old = np.dot(F, X)
    Dinc = (
            np.dot(
                np.dot(
                    lin.pinv(
                        np.dot(S_old, S_old.T) / training_length + math.pow(aperture, -2) * np.eye(Nx)
                    ),
                    S_old
                ),
                Td.T
            ) / training_length
    ).T

    return D + Dinc
# end incremental_loading


# Incremental training
def incremental_training(Wout, X, P, A, ridge_param, training_length, reservoir_size):
    """
    Incremental training of output weights.
    :param Wout: Current Wout
    :param X: Training states.
    :param P: Training pattern.
    :param ridge_param: Regularisation parameter.
    :param training_length: Training length
    :param reservoir_size: Reservoir size
    :return: Incremented Wout
    """
    # Compute Wout
    Tout = P - Wout @ X

    # The linear subspace of the reservoir state space that are not yet
    # occupied by any pattern.
    F = NOT(A)

    # ...
    S = F @ X

    # Compute incremente for Wout
    Wout_inc = (
            (pinv(S @ S.T / training_length + ridge_param * np.eye(Nx)) @ S @ Tout.T) / training_length
    ).T

    # Update Wout
    return Wout + Wout_inc
# end incremental_training
