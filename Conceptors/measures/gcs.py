# -*- coding: utf-8 -*-
#
# File : gcs.py
# Description : Cosine similarities
# Date : 14th of october, 2019
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
import math
import numpy as np
import numpy.linalg as lin


# Generalized cosine similarity
def generalized_cosine_similarity(R1, U1, S1, R2, U2, S2):
    """
    Generalized cosine similarity
    :param R1: Correlation matrix 1 (Nx x Nx)
    :param U1: Singular vectors 1 (Nx x Nx)
    :param S1: Singular values 2 (Nx)
    :param R2: Correlation matrix 2 (Nx x Nx)
    :param U2: Singular vectors 2 (Nx x Nx)
    :param S2: Singular values 2 (Nx)
    :return: Generalized cosine similarity (float) 0 <= similarity <= 1.0
    """
    # Assert types
    assert isinstance(R1, np.ndarray)
    assert isinstance(U1, np.ndarray)
    assert isinstance(S1, np.ndarray)
    assert isinstance(R2, np.ndarray)
    assert isinstance(U2, np.ndarray)
    assert isinstance(S2, np.ndarray)

    # Assert dimensions
    assert R1.ndim == 2
    assert U1.ndim == 2
    assert S1.ndim == 1
    assert R2.ndim == 2
    assert U2.ndim == 2
    assert S2.ndim == 1

    # Square matrix
    assert R1.shape[0] == R1.shape[1]
    assert U1.shape[0] == U1.shape[1]
    assert R2.shape[0] == R2.shape[1]
    assert U2.shape[0] == U2.shape[1]

    # Similarity
    similarity = math.pow(lin.norm(((np.sqrt(np.diag(S1)) @ U1.T) @ U2) @ np.sqrt(np.diag(S2)), 'fro'), 2) / (lin.norm(R1, 'fro') * lin.norm(R2, 'fro'))

    # Assert result
    assert similarity >= 0 and similarity <= float('inf')

    return similarity
# end generalized_cosine_similarity


# Conceptor cosine similarity
def conceptor_cosine_similarity(C1, U1, S1, C2, U2, S2):
    """
    Generalized cosine similarity
    :param C1: Conceptor 1 (Nx x Nx)
    :param U1: Singular vectors 1 (Nx x Nx)
    :param S1: Singular values 1 (Nx)
    :param C2: Conceptor 2 (Nx x Nx)
    :param U2: Singular vectors 2 (Nx x Nx)
    :param S2: Singular values 2 (Nx)
    :return: Conceptor cosine similarity (float) 0 <= similarity <= 1.0
    """
    # Assert types
    assert isinstance(C1, np.ndarray)
    assert isinstance(U1, np.ndarray)
    assert isinstance(S1, np.ndarray)
    assert isinstance(C2, np.ndarray)
    assert isinstance(U2, np.ndarray)
    assert isinstance(S2, np.ndarray)

    # Assert dimensions
    assert C1.ndim == 2
    assert U1.ndim == 2
    assert S1.ndim == 1
    assert C2.ndim == 2
    assert U2.ndim == 2
    assert S2.ndim == 1

    # Square matrix
    assert C1.shape[0] == C1.shape[1]
    assert U1.shape[0] == U1.shape[1]
    assert C2.shape[0] == C2.shape[1]
    assert U2.shape[0] == U2.shape[1]

    # Similarity
    similarity = math.pow(lin.norm(((np.diag(np.sqrt(S1)) @ U1.T) @ U2) @ np.sqrt(np.diag(S2)), 'fro'), 2) / (lin.norm(C1, 'fro') * lin.norm(C2, 'fro'))

    # Assert result
    assert similarity >= 0 and similarity <= float('inf')

    return similarity
# end conceptor_cosine_similarity
