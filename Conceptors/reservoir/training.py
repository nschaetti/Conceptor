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
def ridge_regression(X, Y, ridge_param):
    """
    Ridge regression
    :param X: Samples (Length x Nx OR Nx x length)
    :param Y: Observations (length x Ny OR Ny x length)
    :param ridge_param: Regularization parameter
    :return: Least square solution to Xb + l = Y
    """
    # Assert types
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(ridge_param, float) or isinstance(ridge_param, int)

    # Dimension and values
    assert X.ndim == 2
    assert Y.ndim == 2

    # Size
    n_predictors = X.shape[0]
    assert X.shape[1] == Y.shape[1]

    # Compute solution
    B = (lin.inv(X @ X.T + ridge_param * np.eye(n_predictors)) @ X @ Y.T).T

    return B
# end ridge_regression


# Reservoir loading by states prediction.
def loading(X, Xold, Wbias, ridge_param):
    """
    Reservoir loading by states prediction.
    :param X: Reservoir states (Length x Nx OR Nx x length)
    :param Xold: Timeshifted stated (-1)  (Length x Nx OR Nx x length)
    :param Wbias: Bias matrix (Nx)
    :param ridge_param: Regularization parameter
    :return: Loaded internal weight matrix (Nx x Nx), learned states (
    """
    # Assert types
    assert isinstance(X, np.ndarray)
    assert isinstance(Xold, np.ndarray)
    assert isinstance(Wbias, np.ndarray)
    assert isinstance(ridge_param, float) or isinstance(ridge_param, int)

    # Dimension
    assert X.ndim == 2
    assert Xold.ndim == 2
    assert Wbias.ndim == 1
    assert X.shape[0] == Xold.shape[0]
    assert X.shape[1] == Xold.shape[1]

    # Reservoir size
    learn_length = X.shape[1]
    reservoir_size = X.shape[0]

    # Target for learning W
    X_new = (
        np.arctanh(X) - np.repeat(Wbias.reshape(1, -1), learn_length, axis=0).T
    )
    sample_learn_length = int(learn_length / 4)
    for i in range(4):
        np.save("Y{}.npy".format(i), X_new[:, i*sample_learn_length:i*sample_learn_length+sample_learn_length].T)
    # end for
    np.save("xTx.npy", Xold @ Xold.T)
    np.save("xTy.npy", Xold @ X_new.T)
    np.save("ridge_xTx.npy", Xold @ Xold.T + ridge_param * np.eye(reservoir_size))
    np.save("inv_xTx.npy", lin.inv(Xold @ Xold.T + ridge_param * np.eye(reservoir_size)))
    np.save("W.npy", ((lin.inv(Xold @ Xold.T + ridge_param * np.eye(reservoir_size)) @ Xold) @ X_new.T).T)
    # Learning W
    return (
        (lin.inv(Xold @ Xold.T + ridge_param * np.eye(reservoir_size)) @ Xold) @ X_new.T
    ).T, X_new
# end loading


# Train output weights
def train_outputs(training_states, training_targets, ridge_param_wout):
    """
    Train ESN output weights
    :param training_states: Training states (length x Nx OR Nx x length)
    :param training_targets: Training targets (length x Ny OR Ny x length)
    :param ridge_param_wout: Regularization parameter
    :return: Trained Wout matrix (Ny x Ny)
    """
    # Assert types
    assert isinstance(training_states, np.ndarray)
    assert isinstance(training_targets, np.ndarray)
    assert isinstance(ridge_param_wout, float) or isinstance(ridge_param_wout, int)

    # Dimension
    assert training_states.ndim == 2
    assert training_targets.ndim == 2

    # Solve Ax = b problem
    return ridge_regression(training_states, training_targets, ridge_param_wout)
# end train_outputs


# Incremental loading of a reservoir
def incremental_loading(D, Xold, P, Win, A, aperture, training_length):
    """
    Incremental loading of a reservoir.
    NOT YET TESTED.
    :param D: Current input simulation matrix (Nx x Nx)
    :param Xold: Training states (Nx x Length)
    :param P: Original pattern (1 x Length)
    :param Win: Input weight matrix
    :param A: Current disjonction of all conceptors loaded
    :param training_length: Training length
    :return: The new input simulation matrix D
    """
    # Reservoir size
    reservoir_size = Xold.shape[0]

    # Difference between target patterns and was is predicted by
    Td = Win @ P.reshape(1, -1) - D @ Xold

    # The linear subspace of the reservoir state space that are not yet
    # occupied by any pattern.
    F = NOT(A)

    # Filter old state to get only what is new
    S_old = F @ Xold

    # xTx
    sTs = S_old @ S_old.T / training_length

    # xTx
    sTs_ridge = sTs + math.pow(aperture, -2) * np.eye(reservoir_size)

    # xTy
    sTy = S_old @ Td.T / training_length

    # Compute the increment for matrix D
    Dinc = (lin.pinv(sTs_ridge) @ sTy).T
    """Dinc = (
            np.dot(
                np.dot(
                    lin.pinv(
                        np.dot(S_old, S_old.T) / training_length + math.pow(aperture, -2) * np.eye(reservoir_size)
                    ),
                    S_old
                ),
                Td.T
            ) / training_length
    ).T"""

    return D + Dinc
# end incremental_loading


# Incremental training
def incremental_training(Wout, X, P, A, ridge_param, training_length):
    """
    Incremental training of output weights.
    NOT YET TESTED.
    :param Wout: Current Wout (Ny x Nx)
    :param X: Training states (Nx x Length).
    :param P: Training pattern (1 x Length).
    :param ridge_param: Regularisation parameter.
    :param training_length: Training length
    :return: Incremented Wout
    """
    # Reservoir size
    reservoir_size = X.shape[0]

    # Compute Wout
    Tout = P - Wout @ X

    # The linear subspace of the reservoir state space that are not yet
    # occupied by any pattern.
    F = NOT(A)

    # Filter training states to get what is new
    S = F @ X

    # Compute increment for Wout
    Wout_inc = (
            (pinv(S @ S.T / training_length + ridge_param * np.eye(reservoir_size)) @ S @ Tout.T) / training_length
    ).T

    # Update Wout
    return Wout + Wout_inc
# end incremental_training
