import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load your PCA results
eigenvectors = np.load('pca_results_eigenvectors.npy')
eigenvalues = np.load('pca_results_eigenvalues.npy')
projected_traj = np.load('pca_results_projected_traj.npy')

# Setup plot parameters
plt.figure(figsize=(10, 10))
num_pca_components_to_graph = 2
AX = gridspec.GridSpec(num_pca_components_to_graph, 1)
plot_array = []
for i in range(num_pca_components_to_graph):
    plot_array.append(plt.subplot2grid((num_pca_components_to_graph, 1), (i, 0)))

# Plot eigenvectors or projections
for i in range(num_pca_components_to_graph):
    # Example plotting eigenvectors (replace with your data)
    eigenvector = eigenvectors[i]  # Replace with appropriate eigenvector data
    plot_array[i].plot(eigenvector, label=f'PC {i+1} Eigenvector')

    # Set labels and limits
    plot_array[i].set_xlabel('Feature Index')
    plot_array[i].set_ylabel(f'PC {i+1} Direction')
    plot_array[i].legend()

# Finalize and display plot
plt.tight_layout()
plt.show()

# Save the plot as a PDF
plt.savefig('pca_eigenvectors.pdf')

