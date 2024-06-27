import mdtraj as md
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
 
# Load the trajectory and topology files
traj = md.load('wt_nowat_1us.xtc', top='wt_protein_glycan.pdb')

# Calculate radius of gyration (RG) for each frame
rg_values = md.compute_rg(traj)

# Extract time in nanoseconds
time_ns = traj.time / 1000.0  # Convert picoseconds to nanoseconds

# Plot RG vs Time
plt.figure(figsize=(10, 6))
plt.plot(time_ns, rg_values, label='Radius of Gyration (RG) WT NRXN')
plt.xlabel('Time (ns)')
plt.ylabel('Radius of Gyration (nm)')
plt.title('Radius of Gyration (RG) vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('RG_T.png')
plt.show()

# Calculate the density of RG values
density = gaussian_kde(rg_values.flatten())
rg_range = np.linspace(rg_values.min(), rg_values.max(), 100)
density_values = density(rg_range)

# Plot probability density vs RG
plt.figure(figsize=(10, 6))
plt.plot(rg_range, density_values, label='Probability Density')
plt.fill_between(rg_range, density_values, alpha=0.3)
plt.xlabel('Radius of Gyration (nm)')
plt.ylabel('Probability Density')
plt.title('Probability Density of Radius of Gyration(WT NRXN)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('RG_PD.png')
plt.show()
