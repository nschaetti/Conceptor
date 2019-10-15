# -*- coding: utf-8 -*-
#
# File : gcs.py
# Description : Generalized cosine similarities
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
    :param R1: Correlation matrix 1 (reservoir size x reservoir size)
    :param U1: Singular vectors 1
    :param S1: Singular values 2
    :param R2: Correlation matrix 2
    :param U2: Singular vectors 2
    :param S2: Singular values 2
    :return: Generalized cosine similarity
    """
    # Similarity
    return math.pow(lin.norm(((np.sqrt(np.diag(S1)) @ U1.T) @ U2) @ np.sqrt(np.diag(S2)), 'fro'), 2) / (lin.norm(R1, 'fro') * lin.norm(R2, 'fro'))
# end generalized_cosine_similarity


# Conceptor cosine similarity
def conceptor_cosine_similarity(C1, U1, S1, C2, U2, S2):
    """
    Generalized cosine similarity
    :param C1: Conceptor 1
    :param U1: Singular vectors 1
    :param S1: Singular values 1
    :param C2: Conceptor 2
    :param U2: Singular vectors 2
    :param S2: Singular values 2
    :return: Generalized cosine similarity
    """
    return math.pow(lin.norm(((np.diag(np.sqrt(S1)) @ U1.T) @ U2) @ np.sqrt(np.diag(S2)), 'fro'), 2) / (lin.norm(C1, 'fro') * lin.norm(C2, 'fro'))
# end conceptor_cosine_similarity
