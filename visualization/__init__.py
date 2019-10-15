# coding=utf-8
#
# File : __init__.py
# Description : Visualization sub-package.
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
from .pattern_visualization import plot_patterns_with_singular_values, plot_patterns_with_infos
from .similarities import plot_similarity_matrix
from .plotting import compare_singular_values, plot_singular_values

# ALL
__all__ = ['plot_patterns_with_singular_values', 'plot_similarity_matrix', 'compare_singular_values',
           'plot_singular_values', 'plot_patterns_with_infos']
