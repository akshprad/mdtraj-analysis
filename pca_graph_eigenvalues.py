import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

# Load PCA results
eigenvectors = np.load('pca_results_eigenvectors.npy')  # Eigenvectors
eigenvalues = np.load('pca_results_eigenvalues.npy')    # Eigenvalues
mean_structure = np.load('pca_results_mean.npy')        # Mean structure
std_structure = np.load('pca_results_std.npy')          # Standard deviations
projected_traj = np.load('pca_results_projected_traj.npy')  # Projected trajectory

# Plot eigenvalues
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues**2 / np.sum(eigenvalues**2), alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(eigenvalues) + 1), np.cumsum(eigenvalues**2 / np.sum(eigenvalues**2)), where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Example for plotting the first two principal components against each other
plt.figure(figsize=(10, 6))
plt.scatter(projected_traj[:, 0], projected_traj[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projected Trajectory onto PC1 vs PC2')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_mda.png')
plt.show()

# Load mean structure PDB file using MDAnalysis
mean_universe = mda.Universe('pca_results_mean.pdb')
# Visualize mean structure
#mean_universe.atoms.draw()
import py3Dmol

import nglview as nv

# Load mean structure PDB file using MDAnalysis
mean_universe = mda.Universe('pca_results_mean.pdb')

# Get the mean structure coordinates
coords = mean_universe.atoms.positions

# Create a viewer instance
view = nv.show_mdanalysis(mean_universe)

# Add the mean structure to the viewer
view.add_ball_and_stick()

# Display the viewer
view.center()
view.display()

# Plot eigenvectors (for example, first two components)
plt.figure(figsize=(10, 6))
plt.plot(mean_structure + eigenvectors[0] * std_structure[0], label='PC1')
plt.plot(mean_structure + eigenvectors[1] * std_structure[1], label='PC2')
plt.xlabel('Atom Index')
plt.ylabel('Fluctuation')
plt.title('Principal Components - Directions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_mda2_eigen.png')
plt.show()

