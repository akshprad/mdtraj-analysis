import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np

# Load production trajectory
production_traj = md.load('wt_nowat_1us.xtc', top='wt_protein_glycan.pdb')
print(type(production_traj))

alpha_carbon_indices = production_traj.top.select_atom_indices('alpha')

ref_structure = production_traj[0]
# Check the type and dimensions of ref_structure
print(type(ref_structure))
print(ref_structure)

rmsd_production = md.rmsd(production_traj, ref_structure, atom_indices=alpha_carbon_indices)


np.savetxt('rmsd_production.txt', np.column_stack((production_traj.time, rmsd_production)), header='Time (ps)  RMSD (nm)', fmt='%.6f')

time_production, rmsd_production = np.loadtxt('rmsd_production.txt', unpack=True)

# Plot RMSD for production trajectory only
plt.plot(time_production, rmsd_production, label='Production RMSD', color='red')
plt.xlabel('Time (ps)')
plt.ylabel('RMSD (nm)')
plt.title('Production RMSD_WT')
plt.legend()
plt.savefig('rmsd_production_WT_nrxn.png')
plt.show()



