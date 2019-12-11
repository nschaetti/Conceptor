
# Imports
import numpy as np
import matplotlib.pyplot as plt


# Plot similarity matrix
def plot_similarity_matrix(similarity_matrix, title):
    """
    Plot similarity matrix
    :param similarity_matrix: Similarity matrix
    :param title: Plot's title
    """
    # Show similarity matrices
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(similarity_matrix, interpolation='nearest', cmap='Greys_r')
    plt.title(title)
    fig.colorbar(cax, ticks=np.arange(0.1, 1.1, 0.1))
    for (i, j), z in np.ndenumerate(similarity_matrix):
        if (i < 2 and j < 2) or (i > 1 and j > 1):
            plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        else:
            plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color='white')
        # end if
    # end for
    plt.show()
# end plot_similarity_matrix

