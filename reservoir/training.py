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
from Conceptors.logic import NOT
from numpy.linalg import pinv
import math


# Ridge regression
def ridge_regression(X, Y, ridge_param, dim=0):
    """
    Ridge regression
    :param X: Samples (Length x Nx OR Nx x length)
    :param Y: Observations (length x Ny OR Ny x length)
    :param ridge_param: Regularization parameter
    :param dim: Position of the temporal dimension.
    :return: Least square solution to Xb + l = Y
    """
    # Assert types
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(ridge_param, float) or isinstance(ridge_param, int)
    assert isinstance(dim, int)

    # Dimension and values
    assert X.ndim == 2
    assert Y.ndim == 2
    assert dim == 0 or dim == 1

    # Size
    if dim == 0:
        n_predictors = X.shape[1]
        assert X.shape[0] == Y.shape[0]
    else:
        n_predictors = X.shape[0]
        assert X.shape[1] == Y.shape[1]
    # end if

    # Compute solution
    if dim == 0:
        B = lin.inv(X.T @ X + ridge_param * np.eye(n_predictors)) @ X @ Y
    else:
        B = (lin.inv(X @ X.T + ridge_param * np.eye(n_predictors)) @ X @ Y.T).T
    # end if

    return B
# end ridge_regression


# Reservoir loading by states prediction.
def loading(X, Xold, Wbias, ridge_param, dim=0):
    """
    Reservoir loading by states prediction.
    :param X: Reservoir states (Length x Nx OR Nx x length)
    :param Xold: Timeshifted stated (-1)  (Length x Nx OR Nx x length)
    :param Wbias: Bias matrix (Nx)
    :param ridge_param: Regularization parameter
    :param dim: Position of the temporal dimension.
    :return: Loaded internal weight matrix (Nx x Nx), learned states (
    """
    # Assert types
    assert isinstance(X, np.ndarray)
    assert isinstance(Xold, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(ridge_param, float) or isinstance(ridge_param, int)
    assert isinstance(dim, int)

    # Dimension
    assert X.ndim == 2
    assert Xold.ndim == 2
    assert Wbias.ndim == 1
    assert X.shape[0] == Xold.shape[0]
    assert X.shape[1] == Xold.shape[1]

    # Reservoir size
    if dim == 0:
        learn_length = X.shape[0]
        reservoir_size = X.shape[1]
    else:
        learn_length = X.shape[1]
        reservoir_size = X.shape[0]
    # end if

    # Target for learning W
    if dim == 0:
        X_new = (
                np.arctanh(X) - np.repeat(Wbias.reshape(-1, 1), learn_length, axis=1)
        )
    else:
        X_new = (
                np.arctanh(X) - np.repeat(Wbias.reshape(1, -1), learn_length, axis=0).T
        )
    # end if

    # Learning W
    return (
        (lin.inv(Xold @ Xold.T + ridge_param * np.eye(reservoir_size)) @ Xold) @ X_new.T
    ).T, X_new
# end loading


# Train output weights
def train_outputs(training_states, training_targets, ridge_param_wout, dim=0):
    """
    Train ESN output weights
    :param training_states: Training states (length x Nx OR Nx x length)
    :param training_targets: Training targets (length x Ny OR Ny x length)
    :param ridge_param_wout: Regularization parameter
    :param dim: Position of the temporal dimension (0 or 1)
    :return: Trained Wout matrix (Ny x Ny)
    """
    # Assert types
    assert isinstance(training_states, np.ndarray)
    assert isinstance(training_targets, np.ndarray)
    assert isinstance(ridge_param_wout, float) or isinstance(ridge_param_wout, int)
    assert isinstance(dim, int)

    # Dimension
    assert training_states.ndim == 2
    assert training_targets.ndim == 2
    assert dim == 0 or dim == 1

    # Solve Ax = b problem
    return ridge_regression(training_states, training_targets, ridge_param_wout, dim)
# end train_outputs


# Incremental loading of a reservoir
def incremental_loading(D, X, P, Win, A, aperture, training_length, dim=0):
    """
    Incremental loading of a reservoir.
    NOT YET TESTED.
    :param D: Current input simulation matrix (Nx x Nx)
    :param X: Training states (??)
    :param P: Original pattern (??)
    :param Win: Input weight matrix
    :param A: Current disjonction of all conceptors loaded
    :param training_length: Training length
    :param dim: Position of the temporal dimension (0 or 1)
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
    NOT YET TESTED.
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
