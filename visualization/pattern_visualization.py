

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Plot original and generated patterns with singular values and reservoir states
def plot_patterns_with_singular_values(truth_patterns, generated_patterns, Xs, SVs, color_truth='b', color_generated='r', title='p and y'):
    """
    Plot original and generated patterns with singular values and reservoir states.
    :param truth_patterns: Original patterns
    :param generated_patterns: Generated patterns
    :param Xs: Reservoir states
    :param SVs: Singular values
    :param color_truth: Color line of the original signal
    :param color_generated: Color line of the generated signal
    :param title: Plot title
    """
    # N. patterns
    n_patterns = truth_patterns.shape[0]

    # Figure (square size)
    plt.figure(figsize=(16, 2 * n_patterns))

    # For each pattern
    for p in range(n_patterns):
        # Plot 1 : original pattern and recreated pattern
        plt.subplot(n_patterns, 4, p * 4 + 1)
        plt.plot(generated_patterns[p], color=color_generated, linewidth=5)
        plt.plot(truth_patterns[p], color=color_truth, linewidth=1.5)

        # Title
        if p == 0:
            plt.title(title)
        # end if

        # X labels
        if p == 3:
            plt.xticks([0, 10, 20])
        else:
            plt.xticks([])
        # end if

        # Y limits
        plt.ylim([-1, 1])
        plt.yticks([-1, 0, 1])

        # Plot 2 : states
        plt.subplot(n_patterns, 4, p * 4 + 2)
        plt.plot(Xs[p, 0])
        plt.plot(Xs[p, 1])
        plt.plot(Xs[p, 2])

        # Title
        if p == 0:
            plt.title(u'two neurons')
        # end if

        # X labels
        if p == 3:
            plt.xticks([0, 10, 20])
        else:
            plt.xticks([])
        # end if

        # Y limits
        plt.ylim([-1, 1])
        plt.yticks([-1, 0, 1])

        # Plot 3 : Log10 of singular values (PC energy)
        plt.subplot(n_patterns, 4, p * 4 + 3)
        plt.plot(np.log10(SVs[p]), 'red', linewidth=2)

        # X labels
        if p == 3:
            plt.xticks([0, 50, 100])
        else:
            plt.xticks([])
        # end if

        # Limits
        plt.ylim([-20, 10])

        # Title
        if p == 0:
            plt.title(u'log10 PC Energy')
        # end if

        # Learning PC energy
        plt.subplot(n_patterns, 4, p * 4 + 4)
        plt.plot(SVs[p, :10], 'red', linewidth=2)

        # Title
        if p == 0:
            plt.title(u"leading PC energy")
        # end ifS

        # Limits
        plt.ylim([0, 40.0])
    # end for

    # Show figure
    plt.show()
# end plot_patterns_with_singular_values
