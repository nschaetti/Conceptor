# coding=utf-8
#
# File : test_subspace_demo.py
# Description : Test the subspace demo.
# Date : 15th of October, 2019
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
from unittest import TestCase
import os
import numpy as np
import numpy.linalg as lin
from Conceptors.patterns import patts
import Conceptors.logic


# Test the subspace demo.
class Test_Subspace_Demo(TestCase):
    """
    Test the subspace demo.
    """

    ##############################
    # PRIVATE
    ##############################

    # Subspace demo
    def _subspace_demo(self):
        """
        Subspace demo
        :return:
        """
        # Package
        subpackage_dir, this_filename = os.path.split(__file__)
        package_dir = os.path.join(subpackage_dir, "..")
        DATA_PATH = os.path.join(package_dir, "data", "params")

        # Experiment control
        seed = 1
        reservoir_size = 100

        # Weights scaling
        spectral_radius = 1.5
        input_scaling = 1.5
        bias_scaling = 0.2

        # Sequence lengths
        washout_length = 500
        learn_length = 1000
        signal_plot_length = 20
        conceptor_test_length = 200

        # Regularization
        ridge_param_wstar = 0.0001
        ridge_param_wout = 0.01

        # Aperture
        alpha = 10

        # Apertures to run
        alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]

        # Setting patterns
        patterns = [52, 53, 9, 35]

        # Initialise random generator
        np.random.seed(seed)

        # Load internal weights from matlab file
        Wstar_raw = Conceptors.reservoir.load_matlab_file(os.path.join(DATA_PATH, "WstarRaw.mat"), "WstarRaw")

        # Load input weights from matlab file
        Win_raw = Conceptors.reservoir.load_matlab_file(os.path.join(DATA_PATH, "WinRaw.mat"), "WinRaw")

        # Load Wbias from matlab from
        Wbias_raw = Conceptors.reservoir.load_matlab_file(os.path.join(DATA_PATH, "WbiasRaw"), "WbiasRaw").reshape(-1)

        # Scale raw weights and initialize weights
        Wstar, Win, Wbias = Conceptors.reservoir.scale_weights(
            W=Wstar_raw,
            Win=Win_raw,
            Wbias=Wbias_raw,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling
        )

        # Number of patterns
        n_patterns = len(patterns)

        # We will save all states
        training_states = np.zeros((reservoir_size, n_patterns * learn_length))
        training_states_old = np.zeros((reservoir_size, n_patterns * learn_length))
        training_win_u = np.zeros((reservoir_size, n_patterns * learn_length))
        training_targets = np.zeros((1, n_patterns * learn_length))

        # Save all patterns
        pattern_collectors = np.zeros((n_patterns, learn_length))

        # Save states
        state_collectors = np.zeros((n_patterns, reservoir_size, learn_length))

        # Save singular values
        singular_value_collectors = np.zeros((n_patterns, reservoir_size))

        # State and pattern for plotting
        plotting_states = np.zeros((n_patterns, reservoir_size, signal_plot_length))
        plotting_patterns = np.zeros((n_patterns, signal_plot_length))

        # Last states
        last_states = np.zeros((reservoir_size, n_patterns))

        # Collection of conceptors
        conceptor_collector = Conceptors.conceptors.Collector(reservoir_size)

        # We run the ESN model and we save
        # all the needed informations.
        # For each patterns
        for p in range(n_patterns):
            # Select the right pattern
            patt = patts[patterns[p]]

            # Starting state (x_0 = 0)
            x = np.zeros(reservoir_size)

            # Run reservoir
            state_collector, pattern_collector = Conceptors.reservoir.run(
                pattern=patt,
                reservoir_size=reservoir_size,
                x_start=x,
                Wstar=Wstar,
                Win=Win,
                Wbias=Wbias,
                training_length=learn_length,
                washout_length=washout_length
            )

            # State collector
            state_collectors[p] = state_collector

            # Time shifted states
            state_collector_old = np.zeros((reservoir_size, learn_length))
            state_collector_old[:, 1:] = state_collector[:, :-1]

            # Save last state
            last_states[:, p] = state_collector[:, -1]

            # Train conceptor
            C, U, Snew, Sorg, R = Conceptors.conceptors.train(
                X=state_collector,
                aperture=alpha,
                dim=1
            )

            # Save conceptor and singular values
            conceptor_collector.add(p, C, U, Snew, Sorg, R)
            singular_value_collectors[p] = Sorg

            # Plotting states and patterns
            plotting_states[p] = state_collector[:, :signal_plot_length]
            plotting_patterns[p] = pattern_collector[:signal_plot_length]

            # Save patterns
            pattern_collectors[p] = pattern_collector

            # Save states for training
            training_states[:, p * learn_length:(p + 1) * learn_length] = state_collector
            training_states_old[:, p * learn_length:(p + 1) * learn_length] = state_collector_old
            training_targets[0, p * learn_length:(p + 1) * learn_length] = pattern_collector
            training_win_u[:, p * learn_length:(p + 1) * learn_length] = Win * pattern_collector
        # end for

        # Compute readout
        Wout = Conceptors.reservoir.train_outputs(
            training_states=training_states,
            training_targets=training_targets,
            ridge_param_wout=ridge_param_wout,
            dim=1
        )

        # Compute training error (NRMSE)
        NRMSE_readout = Conceptors.measures.nrmse(np.dot(Wout, training_states), training_targets)

        # Load W
        W, X_new = Conceptors.reservoir.loading(
            X=training_states,
            Xold=training_states_old,
            Wbias=Wbias,
            ridge_param=ridge_param_wstar,
            dim=1
        )

        # Training errors per neuron
        NRMSE_W = Conceptors.measures.nrmse(np.dot(W, training_states_old), X_new)

        # Output states and signals
        conceptor_test_states = np.zeros((n_patterns, conceptor_test_length, reservoir_size))
        conceptor_test_output = np.zeros((n_patterns, conceptor_test_length))

        # Now we generate signal but with conceptors
        # Save states and outputs
        for p in range(n_patterns):
            # Get the corresponding conceptors
            C, _, _, _, _ = conceptor_collector.get(p)

            # Original starting state or get the original
            original_x = Conceptors.reservoir.load_matlab_file(
                os.path.join(DATA_PATH, "{}{}.mat".format("test_C_x", p + 1)),
                "x"
            )
            x = original_x.reshape(reservoir_size)

            # Free run with the loaded reservoir
            test_states, test_output = Conceptors.reservoir.free_run(
                x_start=x,
                W=W,
                Wbias=Wbias,
                Wout=Wout,
                C=C,
                run_length=conceptor_test_length,
                washout_length=0
            )

            # Save
            conceptor_test_states[p] = test_states
            conceptor_test_output[p] = test_output
        # end for

        # Variables for plotting
        interpolation_rate = 20

        # Aligned results
        conceptor_test_output_aligned = np.zeros((n_patterns, signal_plot_length))
        conceptor_test_states_aligned = np.zeros((n_patterns, signal_plot_length, 5))

        # Measured performances
        NRMSE_aligned = np.zeros(n_patterns)
        MSE_aligned = np.zeros(n_patterns)

        # For each pattern
        for p in range(n_patterns):
            # Interpolate and align truth and generated
            aligned_ouputs, max_ind = Conceptors.tools.interpolation_alignment(
                truth_pattern=plotting_patterns[p],
                generated_pattern=conceptor_test_output[p],
                interpolation_rate=interpolation_rate,
                dim=1
            )

            # Get aligned states
            conceptor_test_states_aligned[p] = conceptor_test_states[p, max_ind:max_ind + signal_plot_length, :5]

            # Save aligned outputs
            conceptor_test_output_aligned[p] = aligned_ouputs

            # Evaluate aligned signals with NRMSE
            NRMSE_aligned[p] = Conceptors.measures.nrmse(
                conceptor_test_output_aligned[p].reshape(1, -1),
                plotting_patterns[p].reshape(1, -1)
            )

            # Evaluate aligned signals with MSE
            MSE_aligned[p] = np.average(np.power(conceptor_test_output_aligned[p] - plotting_patterns[p], 2))
        # end for

        # Energy similarities between driven response spaces
        # List of conceptor similarities
        similarity_matrix_conceptors = np.zeros((n_patterns, n_patterns))

        # For each combination of patterns
        for p1 in range(n_patterns):
            for p2 in range(n_patterns):
                # Get conceptors
                C1, U1, S1, _, _ = conceptor_collector.get(p1)
                C2, U2, S2, _, _ = conceptor_collector.get(p2)

                # Cosine similarity on conceptor matrices
                similarity = Conceptors.measures.conceptor_cosine_similarity(
                    C1=C1,
                    U1=U1,
                    S1=S1,
                    C2=C2,
                    U2=U2,
                    S2=S2
                )

                # Save
                similarity_matrix_conceptors[p1, p2] = similarity
            # end for
        # end for

        # List of correlation matrix similarities
        similarity_matrix_correlations = np.zeros((n_patterns, n_patterns))

        # For each combination of patterns
        for p1 in range(n_patterns):
            for p2 in range(n_patterns):
                # Get conceptors
                _, U1, _, S1, R1 = conceptor_collector.get(p1)
                _, U2, _, S2, R2 = conceptor_collector.get(p2)

                # Cosine similarity on correlation matrix
                similarity = Conceptors.measures.generalized_cosine_similarity(
                    R1=R1,
                    U1=U1,
                    S1=S1,
                    R2=R2,
                    U2=U2,
                    S2=S2
                )

                # Save
                similarity_matrix_correlations[p1, p2] = similarity
            # end for
        # end for

        # Save each singular values
        singular_values_sine = np.zeros((len(alphas), reservoir_size))
        singular_values_periodic = np.zeros((len(alphas), reservoir_size))

        # For each alpha value
        for i, a in enumerate(alphas):
            # Get correlation matrices
            _, _, _, _, R1 = conceptor_collector.get(0)
            _, _, _, _, R2 = conceptor_collector.get(2)

            # Change aperture to a
            C1 = np.dot(R1, lin.inv(R1 + np.power(a, -2) * np.eye(reservoir_size)))
            U1, S1, V1 = lin.svd(C1)
            singular_values_sine[i] = S1

            # Change aperture to
            C2 = np.dot(R2, lin.inv(R2 + np.power(a, -2) * np.eye(reservoir_size)))
            U2, S2, V2 = lin.svd(C2)
            singular_values_periodic[i] = S2
        # end for

        return (
            NRMSE_readout, np.average(NRMSE_W), NRMSE_aligned, similarity_matrix_correlations,
            similarity_matrix_conceptors, singular_values_sine, singular_values_periodic
        )
    # end _subspace_demo

    ##############################
    # TESTS
    ##############################

    # Test Wout training
    def test_wout(self):
        """
        Test Wout training
        :return:
        """
        # Run subspace demo
        NRMSE_readout, _, _, _, _, _, _ = self._subspace_demo()

        # Check NRMSE readout
        self.assertAlmostEqual(NRMSE_readout, 0.00067741, places=4, msg="NRMSE readout is not correct")
    # end test_wout

    # Test reservoir loading
    def test_reservoir_loading(self):
        """
        Test reservoir loading
        :return:
        """
        # Run subspace demo
        _, NRMSE_W, _, _, _, _, _ = self._subspace_demo()

        # Check NRMSE W
        self.assertAlmostEqual(NRMSE_W, 0.030043524742344337, places=4, msg="NRMSE W is not correct")
    # end test_reservoir_loading

    # Test pattern generation
    def test_pattern_generation(self):
        """
        Test pattern generation
        :return:
        """
        # Run subspace demo
        _, _, NRMSE_pattern, _, _, _, _ = self._subspace_demo()

        # Size
        self.assertIsInstance(NRMSE_pattern, np.ndarray, msg="Pattern NRMSEs not a Numpy array")
        self.assertEqual(NRMSE_pattern.ndim, 1, msg="Pattern NRMSEs not 1-dim array")
        self.assertEqual(NRMSE_pattern.shape[0], 4, msg="Pattern NRMSE not containing four measures")

        # True NRMSEs
        true_NRMSEs = np.ndarray([0.0079004, 0.00646076, 0.03286722, 0.00665655])

        # Check NRMSE W
        for i in range(NRMSE_pattern.shape[0]):
            self.assertAlmostEqual(
                NRMSE_pattern[i],
                true_NRMSEs[i],
                places=4,
                msg="Pattern NRMSE {} is not correct".format(i+1)
            )
        # end for
    # end test_pattern_generation

    # Test similarities
    def test_similarities(self):
        """
        Test similarities
        :return:
        """
        # Run subspace demo
        _, _, _, R_sim, C_sim, _, _ = self._subspace_demo()

        # Check type
        self.assertIsInstance(R_sim, np.ndarray, msg="R similarity matrix is not a Numpy array")
        self.assertIsInstance(C_sim, np.ndarray, msg="C similarity matrix is not a Numpy array")

        # Check dimension
        self.assertEqual(R_sim.ndim, 2, msg="R similarity matrix dimension is not 2")
        self.assertEqual(C_sim.ndim, 2, msg="C similarity matrix dimension is not 2")

        # Check size
        self.assertEqual(R_sim.shape[0], 4, msg="R similarity matrix dimension 0 is not 4")
        self.assertEqual(R_sim.shape[1], 4, msg="R similarity matrix dimension 1 is not 4")
        self.assertEqual(C_sim.shape[0], 4, msg="C similarity matrix dimension 0 is not 4")
        self.assertEqual(C_sim.shape[1], 4, msg="C similarity matrix dimension 1 is not 4")

        # True R similarities
        true_R_sim = np.array([
            [1.0,        0.98942504, 0.46658877, 0.49367049],
            [0.98942504, 1.,         0.427216,   0.45357372],
            [0.46658877, 0.427216,   1.,         0.96988298],
            [0.49367049, 0.45357372, 0.96988298, 1.]
        ])

        # True C similarities
        true_C_sim = np.array([
            [1.,         0.81706046, 0.29889339, 0.29934575],
            [0.81706046, 1.,         0.30180539, 0.30757618],
            [0.29889339, 0.30180539, 1.,         0.96809685],
            [0.29934575, 0.30757618, 0.96809685, 1.]]
        )

        # Check similarities
        for i in range(4):
            for j in range(4):
                # Check R
                self.assertAlmostEqual(
                    R_sim[i, j],
                    true_R_sim[i, j],
                    places=4,
                    msg="R similarity matrix value ({}, {}) is wrong".format(i, j)
                )

                # Check C
                self.assertAlmostEqual(
                    C_sim[i, j],
                    true_C_sim[i, j],
                    places=4,
                    msg="C similarity matrix value ({}, {}) is wrong".format(i, j)
                )
            # end for
        # end for
    # end test_similarities

    # Test aperture and singular values
    def test_aperture_singular_values(self):
        """
        Test aperture and singular values
        :return:
        """
        # Run subspace demo
        _, _, _, _, _, sine, periodic = self._subspace_demo()

        # Check type
        self.assertIsInstance(sine, np.ndarray, msg="Sine SVs is not a Numpy array")
        self.assertIsInstance(periodic, np.ndarray, msg="Periodic SVs is not a Numpy array")

        # Check dimension
        self.assertEqual(sine.ndim, 2, msg="Sine SVs dimension is not 2")
        self.assertEqual(periodic.ndim, 2, msg="Periodic SVs dimension is not 2")

        # Check size
        self.assertEqual(sine.shape[0], 5, msg="Sine SVs dimension 0 is not 5")
        self.assertEqual(sine.shape[1], 100, msg="Sine SVs dimension 1 is not 100")
        self.assertEqual(periodic.shape[0], 5, msg="Periodic SVs dimension 0 is not 5")
        self.assertEqual(periodic.shape[1], 100, msg="Periodic SVs dimension 1 is not 100")

        # Sine true singular values
        sine_truth = np.array([
            [0.96690364, 0.92937833, 0.7527158,  0.45442216, 0.35322191],
            [0.99965782, 0.9992407,  0.99672553, 0.98813646, 0.98201844],
            [0.99999658, 0.9999924,  0.99996715, 0.99987995, 0.99981693],
            [0.99999997, 0.99999992, 0.99999967, 0.9999988,  0.99999817],
            [1.00000007, 1.00000001, 1.,         0.99999999, 0.99999998]]
        )

        # Periodic true singular valeurs
        periodic_truth = np.array([
            [0.94556989, 0.91813907, 0.88711959, 0.82450062, 0.77085975],
            [0.9994247,  0.9991092,  0.99872918, 0.99787597, 0.99703628],
            [0.99999424, 0.99999108, 0.99998728, 0.99997871, 0.99997028],
            [0.99999994, 0.99999991, 0.99999987, 0.99999979, 0.9999997 ],
            [1.00000005, 1.00000001, 0.99999999, 0.99999999, 0.99999993]]
        )

        # Test values
        for i in range(5):
            for j in range(5):
                # Assert sine
                self.assertAlmostEqual(
                    sine[i, j],
                    sine_truth[i, j],
                    places=4,
                    msg="Wrong value for sine SVs at ({}, {})".format(i, j)
                )

                # Assert periodic
                self.assertAlmostEqual(
                    periodic[i, j],
                    periodic_truth[i, j],
                    places=4,
                    msg="Wrong value for periodic SVs at ({}, {})".format(i, j)
                )
            # end for
        # end for
    # end test_aperture_singular_values

# end Test_Subspace_Demo
