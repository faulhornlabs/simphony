# a script to produce the lattice indices and hyperfine tensor pairs, running time is around 30 s
# lattice indices: v = n1 * a1 + n2 * a2 + n3 * a3 + n4 * tau (n1, n2, n3 are integers, n4 is binary)
# a1 = aCC*{0, 2*sqrt(2)/3, 4/3}, a2 = aCC*{-sqrt(6)/3, -sqrt(2)/3, 4/3}, a3 = aCC*{sqrt(6)/3, -sqrt(2)/3, 4/3}, tau = (a1 + a2 + a3) / 4 = aCC*{0,0,1}, aCC = sqrt(3)/4*3.567 = 1.545
# usage: python nv_hyperfine_database_lattice_index.py nv-2.txt nv_hyperfine_database.csv

import sys

import numpy as np
import pandas as pd
import scipy

if len(sys.argv) != 3:
    print("Usage: python nv_hyperfine_database_lattice_index.py input-file output-file")
    sys.exit(1)

input_file = str(sys.argv[1])
output_file = str(sys.argv[2])
NITROGEN_ROW = 0
VACANCY_ROW = 4

# import the dft lattice
names = ['index', 'distance_from_nitrogen_dft', 'x_dft', 'y_dft', 'z_dft', 'Axx', 'Ayy', 'Azz', 'Axy', 'Axz', 'Ayz']
points_dft = pd.read_csv(input_file, sep=r'\s+', names=names, engine='python')
points_dft = points_dft.sort_values('distance_from_nitrogen_dft')
points_dft.reset_index(drop=True, inplace=True)

# The current generator assumes the sorted nv-2.txt layout where the nitrogen
# and vacancy entries appear at fixed row indices.
points_dft['type'] = 'C'
points_dft.loc[NITROGEN_ROW, 'type'] = 'N'
points_dft.loc[VACANCY_ROW, 'type'] = 'V'
points_dft.z_dft -= points_dft.loc[VACANCY_ROW, 'z_dft']

pos_N_dft = np.array(points_dft[points_dft.type == 'N'][['x_dft', 'y_dft', 'z_dft']])[0]
pos_V_dft = np.array(points_dft[points_dft.type == 'V'][['x_dft', 'y_dft', 'z_dft']])[0]
pos_center_dft = (pos_N_dft + pos_V_dft) / 2
points_dft['distance_from_center'] = np.sqrt(
    (points_dft.x_dft - pos_center_dft[0]) ** 2
    + (points_dft.y_dft - pos_center_dft[1]) ** 2
    + (points_dft.z_dft - pos_center_dft[2]) ** 2
)

# produce the original/ideal lattice
rotation = scipy.spatial.transform.Rotation.from_euler(
    seq='zyz',
    angles=[3 * np.pi / 4, np.arccos(1 / np.sqrt(3)), 5 * np.pi / 6],
)
R = rotation.as_matrix()

d = 3.567
a1 = R @ np.array([0, d / 2, d / 2])
a2 = R @ np.array([d / 2, 0, d / 2])
a3 = R @ np.array([d / 2, d / 2, 0])
tau = (a1 + a2 + a3) / 4

pos_center_lattice = tau / 2
lattice_radius_cutoff = 1.05 * float(points_dft.distance_from_center.max())
A = np.column_stack([a1, a2, a3])
inverse_a_inf_norm = np.linalg.norm(np.linalg.inv(A), ord=np.inf)
n_cut = int(np.ceil(inverse_a_inf_norm * (lattice_radius_cutoff + np.linalg.norm(tau) / 2))) + 1
rows = []
for n1 in range(-n_cut, n_cut + 1):
    for n2 in range(-n_cut, n_cut + 1):
        for n3 in range(-n_cut, n_cut + 1):
            for n4 in [0, 1]:
                vec = n1 * a1 + n2 * a2 + n3 * a3 + n4 * tau
                new_row = {
                    'x_lattice': vec[0],
                    'y_lattice': vec[1],
                    'z_lattice': vec[2],
                    'n1': n1,
                    'n2': n2,
                    'n3': n3,
                    'n4': n4,
                }
                rows.append(new_row)

points_ideal = pd.DataFrame(rows)

points_ideal['distance_from_lattice_center'] = np.sqrt(
    (points_ideal.x_lattice - pos_center_lattice[0]) ** 2
    + (points_ideal.y_lattice - pos_center_lattice[1]) ** 2
    + (points_ideal.z_lattice - pos_center_lattice[2]) ** 2
)

points_ideal = points_ideal[points_ideal.distance_from_lattice_center <= lattice_radius_cutoff]
points_ideal = points_ideal.reset_index(drop=True)


# match the two lattices
def find_closest_points(df1, df2):
    closest_points = []
    for _, row1 in df1.iterrows():
        distances = np.sqrt(
            (df2['x_lattice'] - row1['x_dft']) ** 2
            + (df2['y_lattice'] - row1['y_dft']) ** 2
            + (df2['z_lattice'] - row1['z_dft']) ** 2
        )
        closest_index = distances.idxmin()
        closest_points.append(df2.loc[closest_index])
    return pd.DataFrame(closest_points)


closest_points_df = find_closest_points(points_dft, points_ideal)
points_dft[['x_lattice', 'y_lattice', 'z_lattice', 'distance_from_lattice_center', 'n1', 'n2', 'n3', 'n4']] = closest_points_df[
    ['x_lattice', 'y_lattice', 'z_lattice', 'distance_from_lattice_center', 'n1', 'n2', 'n3', 'n4']
].values
points_dft['displacement'] = np.sqrt(
    (points_dft.x_dft - points_dft.x_lattice) ** 2
    + (points_dft.y_dft - points_dft.y_lattice) ** 2
    + (points_dft.z_dft - points_dft.z_lattice) ** 2
)

duplicated_lattice_sites = points_dft.duplicated(subset=['n1', 'n2', 'n3', 'n4'])
if duplicated_lattice_sites.any():
    raise RuntimeError('Closest-point matching produced duplicated lattice indices.')

for n in ['n1', 'n2', 'n3', 'n4']:
    points_dft[n] = points_dft[n].astype('int')

points_dft[['z_dft', 'z_lattice']] -= np.linalg.norm(tau) / 2
points_dft = points_dft.drop(columns=['index', 'distance_from_nitrogen_dft', 'distance_from_center'])
points_dft = points_dft.sort_values('distance_from_lattice_center')
points_dft = points_dft.reset_index(drop=True)

tensor_columns = ['Axx', 'Ayy', 'Azz', 'Axy', 'Axz', 'Ayz']
points_dft.loc[points_dft['type'] != 'C', tensor_columns] = np.nan

points_dft = points_dft[
    [
        'type',
        'n1',
        'n2',
        'n3',
        'n4',
        'x_lattice',
        'y_lattice',
        'z_lattice',
        'x_dft',
        'y_dft',
        'z_dft',
        'displacement',
        'distance_from_lattice_center',
        'Axx',
        'Ayy',
        'Azz',
        'Axy',
        'Axz',
        'Ayz',
    ]
]

# export the indices-hyperfine tensors to csv
points_dft.to_csv(output_file, index=False)
