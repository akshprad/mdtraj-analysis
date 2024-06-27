import numpy as np
import shutil
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
import uuid
import os
from sklearn.decomposition import PCA
import argparse
import warnings
warnings.filterwarnings('ignore')

# Define file paths for trajectories and topologies
traj_files = ['d140y_nl_nowat.xtc', 'p89l_nl_nowat.xtc', 'wt_nlgn_nowat.xtc']
top_files = ['d140y_prot_glyc.pdb', 'p89l_prot_glyc.pdb', 'wt_prot_glyc.pdb']

parser = argparse.ArgumentParser()

parser.add_argument('--out', '-o', dest='out_file', default="pca", help='Output file prefix', type=str)
parser.add_argument('--stride', '-dt', dest='stride', default=1, help='Stride through trajectory skipping this many frames.', type=int)
parser.add_argument('--no-std', dest='standardise_bool', action="store_false", help='Do not standardise coordinates. Default behaviour is to standardise the coordinates.')
parser.add_argument('--save-proj', dest='save_projection', action="store_true", help='Save PCA-projected coordinates.', default=False)
parser.add_argument('--n-components', '-n', dest='n_components', default=2, help='Number of Principal Components to calculate.', type=int)
parser.add_argument('--vis-multiply', dest='extend', default=1, help='Multiply the eigenvectors by this magnitude for visualisation purposes.', type=float)
parser.add_argument('--dir-root', dest='dir_root', help='Directory to store intermediate files. Default is a temporary directory.')

parser.add_argument('--reference', dest='reference', help='Reference structure file, used for alignment')
parser.add_argument('--ref-top', dest='ref_top', help='Reference topology file, used for alignment')
parser.add_argument('--symmetry-list', dest='symmetry_list', nargs="+", help='List of selections which compose a single symmetry group.')

parser.add_argument('--selection', '-s', dest='selection_string', help='Selection string for fitting. Default is "name CA".', type=str, default="name CA")
parser.add_argument('--n-frames', '-fn', dest='n_vis_frames', help='Number of frames to visualise', type=int, default=30)
parser.add_argument('--in-mem', dest='in_mem', action="store_true", help='Perform alignment processing in memory.')

args = parser.parse_args()

print(f"Using trajectories: {traj_files}")
print(f"Using topologies: {top_files}")

# Check if dir_root is provided, otherwise use a temporary directory
if args.dir_root is None:
    dir_root = '/tmp/' + str(uuid.uuid4()) + '/'
else:
    dir_root = args.dir_root

print(f"Using directory root: {dir_root}")
os.makedirs(dir_root, exist_ok=True)

# Load trajectories
trajectories = []
for traj_file, top_file in zip(traj_files, top_files):
    traj = mda.Universe(top_file, traj_file)
    trajectories.append(traj)

# Create selection object
selection_object = trajectories[0].select_atoms(args.selection_string)

# Symmetry handling
if args.symmetry_list:
    n_atoms_set = set([selection_object.select_atoms(current_symmetry).n_atoms for current_symmetry in args.symmetry_list])

    if len(n_atoms_set) > 1:
        raise ValueError('Symmetry group selections have atom groups with different numbers of atoms. Check your selections.')
    if list(n_atoms_set)[0] == 0:
        raise ValueError('Symmetry group selection has resulted in no atoms. Check selection string.')
    if selection_object.n_atoms % list(n_atoms_set)[0] > 1:
        raise ValueError('Number of atoms in symmetry group does not evenly divide full selection object. Check selection strings.')

    temp_coords1 = [[selection_object.select_atoms(current_symmetry).positions for ts in traj.trajectory[::args.stride]] for current_symmetry in args.symmetry_list]

    segids = list(set(selection_object.segids))
    segids_indices = selection_object.atoms.segindices

    selection_object.write(dir_root + 'temp_pdb_file.pdb')

    analysis_universe = mda.Universe(dir_root + 'temp_pdb_file.pdb')

    coordinates = np.empty((len(traj.trajectory[::args.stride]) * len(args.symmetry_list), selection_object.n_atoms, 3), dtype=np.float32)

    coordinates[0:len(traj.trajectory[::args.stride])] = np.concatenate(temp_coords1, axis=1)

    for i in range(1, len(args.symmetry_list)):
        temp_coords1 = np.concatenate(([temp_coords1[-1]], temp_coords1[0:-1]), axis=0)
        coordinates[i * len(traj.trajectory[::args.stride]):(i + 1) * len(traj.trajectory[::args.stride])] = np.concatenate(temp_coords1, axis=1)

else:
    selection_object.write(dir_root + 'temp_pdb_file.pdb')

    analysis_universe = mda.Universe(dir_root + 'temp_pdb_file.pdb')
    coordinates = np.array([selection_object.positions for ts in traj.trajectory[::args.stride]])

analysis_universe = analysis_universe.load_new(coordinates)

with mda.Writer((dir_root + 'analysis_universe_traj.dcd'), analysis_universe.atoms.n_atoms) as W:
    for ts in analysis_universe.trajectory:
        W.write(analysis_universe.atoms)

# Load reference structure
if args.reference:
    if args.ref_top:
        ref_universe = mda.Universe(args.ref_top, args.reference)
    else:
        ref_universe = mda.Universe(top_files[0], args.reference)
else:
    ref_universe = mda.Universe(dir_root + 'temp_pdb_file.pdb')
    ref_universe.load_new(coordinates)

analysis_universe = mda.Universe(dir_root + 'temp_pdb_file.pdb')
analysis_universe.load_new(dir_root + 'analysis_universe_traj.dcd')

# Perform alignment
print('Aligning Trajectory')
if args.in_mem:
    aligner = align.AlignTraj(analysis_universe, ref_universe, filename=dir_root + 'aligned.dcd', select=args.selection_string, verbose=True).run()
    analysis_universe = analysis_universe.load_new(dir_root + 'aligned.dcd')
    aligner = align.AlignTraj(analysis_universe, analysis_universe, filename=dir_root + 'aligned2.dcd', select=args.selection_string, verbose=True).run()
    os.remove(dir_root + 'aligned.dcd')
    analysis_universe = analysis_universe.load_new(dir_root + 'aligned2.dcd')
else:
    aligner = align.AlignTraj(analysis_universe, ref_universe, filename=dir_root + 'aligned.dcd', select=args.selection_string, verbose=True).run()
    analysis_universe = analysis_universe.load_new(dir_root + 'aligned.dcd')
    aligner = align.AlignTraj(analysis_universe, analysis_universe, filename=dir_root + 'aligned2.dcd', select=args.selection_string, verbose=True).run()
    analysis_universe = analysis_universe.load_new(dir_root + 'aligned2.dcd')

print('Trajectory Aligned')

# Calculate covariance
analysis_universe_coordinates = np.array([analysis_universe.atoms.positions for ts in analysis_universe.trajectory], dtype=np.float32)
analysis_universe_coordinates = analysis_universe_coordinates.reshape((analysis_universe.trajectory.n_frames, analysis_universe.atoms.n_atoms * 3))

store_mean = np.mean(analysis_universe_coordinates, axis=0)
demeaned_coords = analysis_universe_coordinates - store_mean
demeaned_coords = demeaned_coords.reshape(analysis_universe.trajectory.n_frames, analysis_universe.atoms.n_atoms, 3)
dist_from_mean = np.linalg.norm(demeaned_coords, axis=2)

# Standardise coordinates if requested
if args.standardise_bool:
    print('Using standardised coordinates')
    pca_ready_coords = (analysis_universe_coordinates - store_mean) / np.std(analysis_universe_coordinates, axis=0)
else:
    print('Not using standardised coordinates')
    pca_ready_coords = analysis_universe_coordinates

print('Calculating PCA')
pc = PCA(n_components=args.n_components)
print("First 10 coordinates:", pca_ready_coords[:10])
pc.fit(pca_ready_coords)

# Save PCA results
np.save(args.out_file + '_eigenvectors.npy', pc.components_)
np.save(args.out_file + '_eigenvalues.npy', np.sqrt(pc.explained_variance_))
np.save(args.out_file + '_mean.npy', store_mean)
np.save(args.out_file + '_std.npy', np.std(analysis_universe_coordinates, axis=0))

if args.save_projection:
    np.save(args.out_file + '_projected_traj.npy', pc.transform(pca_ready_coords))

mean_universe = mda.Merge(selection_object)
mean_coords = np.reshape(store_mean, (analysis_universe.atoms.n_atoms, 3))
mean_universe.load_new(mean_coords, order="fac")
mean_universe.atoms.write(args.out_file + '_mean.pdb')

print('Writing output')
for i in range(args.n_components):
    pc_vector = pc.components_[i, :] * np.sqrt(pc.explained_variance_[i])
    origin = np.zeros(len(pc_vector))

    pc_traj = np.linspace(-pc_vector, pc_vector, args.n_vis_frames) * args.extend

    if args.standardise_bool:
        projected = (pc_traj + pc.mean_.flatten()) + store_mean
    else:
        projected = (pc_traj + pc.mean_.flatten())

    coordinates = projected.reshape(len(pc_traj), -1, 3)

    proj1 = mda.Merge(selection_object)
    proj1.load_new(coordinates, order="fac")

    new_sel_object = proj1.select_atoms(args.selection_string)
    new_sel_object.write(args.out_file + str(i) + '.pdb')

    with mda.Writer((args.out_file + str(i) + '.xtc'), new_sel_object.n_atoms) as W:
        for ts in proj1.trajectory:
            W.write(new_sel_object)

if args.dir_root is None:
    shutil.rmtree(dir_root)

