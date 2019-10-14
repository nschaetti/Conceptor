# coding=utf-8
#
# File : __init__.py
# Description : Measure sub-package
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
from .gcs import generalized_cosine_similarity, conceptor_cosine_similarity
from .nrsme import nrmse, nrmse_aligned

# All
__all__ = ['generalized_cosine_similarity', 'conceptor_cosine_similarity', 'nrmse', 'nrmse_aligned']
