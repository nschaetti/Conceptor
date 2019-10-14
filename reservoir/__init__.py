# coding=utf-8
#
# File : __init__.py
# Description : Reservoir and ESN tools
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
from .free_run import free_run, free_run_input_simulation, free_run_input_recreation
from .run import run
from .training import ridge_regression, train_outputs, incremental_loading, incremental_training
from .weights import generate_internal_weights, from_matlab, scale_weights


# All
__all__ = ['free_run', 'free_run_input_simulation', 'free_run_input_recreation', 'generate_internal_weights',
           'incremental_loading', 'incremental_training', 'run', 'ridge_regression', 'train_outputs', 'from_matlab',
           'scale_weights']
