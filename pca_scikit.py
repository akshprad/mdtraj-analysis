import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define file paths for trajectories and topologies
traj_files = ['d140y_nl_nowat.xtc', 'p89l_nl_nowat.xtc', 'wt_nlgn_nowat.xtc']
top_files = ['d140y_prot_glyc.pdb', 'p89l_prot_glyc.pdb', 'wt_prot_glyc.pdb']

# Load and prepare trajectories
trajectories = []
for traj_file, top_file in zip(traj_files, top_files):
    traj = md.load(traj_file, top=top_file)
    # Align to the first frame to remove rotational and translational motions
    traj.superpose(traj, 0)
    # Remove water molecules if necessary
    traj = traj.remove_solvent()
    # Append to list
    trajectories.append(traj)

# Perform PCA on each trajectory
pcas = []
for traj, traj_file in zip(trajectories, traj_files):
    # Calculate RMSD to align frames (if necessary, adjust mask to select appropriate atoms)
    rmsd = md.rmsd(traj, traj, 0)
    
    # Perform PCA on aligned frames
    pca = PCA(n_components=2)
    
    # Fit PCA to XYZ coordinates (remove hydrogens if necessary)
    xyz = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    pca.fit(xyz)
    
    # Transform coordinates to principal components
    pca_coords = pca.transform(xyz)
    
    # Append PCA results to list
    pcas.append(pca_coords)

# Plotting PCA Results
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r']  # Adjust colors for each trajectory: wt, d140y, p89l
labels = ['d140y', 'p89l', 'wt']  # Labels for each trajectory

for i, (pca_coords, label) in enumerate(zip(pcas, labels)):
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], color=colors[i], label=label)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Analysis of MD Trajectories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca.png')
plt.show()

