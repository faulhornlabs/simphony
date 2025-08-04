# a script to produce the lattice indices and hyperfine tensor pairs, running time is around 30 s
# lattice indices: v = n1 * a1 + n2 * n2 + n3 * a3 + n4 * tau (n1, n2, n3 are integers, n4 is binary)
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

# import the dft lattice
names = ['index','distance','x','y','z','Axx','Ayy','Azz','Axy','Axz','Ayz']
points_dft = pd.read_csv(input_file, sep=' ', names=names)
points_dft = points_dft.sort_values('distance')
points_dft.reset_index(drop=True, inplace=True)
points_dft['type'] = 'C'
points_dft.loc[0, 'type'] = 'N'
points_dft.loc[4, 'type'] = 'V'
points_dft.z -= points_dft.loc[4, 'z']

pos_N_dft = np.array(points_dft[points_dft.type == 'N'][['x','y','z']])[0]
pos_V_dft = np.array(points_dft[points_dft.type == 'V'][['x','y','z']])[0]
pos_center_dft = (pos_N_dft + pos_V_dft) / 2
points_dft['distance_from_center'] = np.sqrt( (points_dft.x - pos_center_dft[0]) ** 2 + (points_dft.y - pos_center_dft[1]) ** 2 + (points_dft.z - pos_center_dft[2]) ** 2 )

# produce the original/ideal lattice
rotation = scipy.spatial.transform.Rotation.from_euler(seq='zyz', angles=[3*np.pi/4, np.arccos(1/np.sqrt(3)), 5*np.pi/6])
R = rotation.as_matrix()

d = 3.567
a1 = R @ np.array([0,d/2,d/2])
a2 = R @ np.array([d/2,0,d/2])
a3 = R @ np.array([d/2,d/2,0])
tau = (a1+a2+a3)/4

n_cut = 15
rows = []
for n1 in range(-n_cut,n_cut+1):
    for n2 in range(-n_cut,n_cut+1):
        for n3 in range(-n_cut,n_cut+1):
            for n4 in [0,1]:
                vec = n1*a1 + n2*a2 + n3*a3 + n4*tau
                new_row = {'x': vec[0], 'y': vec[1], 'z': vec[2], 'n1': n1, 'n2': n2, 'n3': n3, 'n4': n4}
                rows.append(new_row)

points_ideal = pd.DataFrame(rows)

pos_center_ideal = tau / 2
points_ideal['distance_from_center'] = np.sqrt( (points_ideal.x - pos_center_ideal[0]) ** 2 + (points_ideal.y - pos_center_ideal[1]) ** 2 + (points_ideal.z - pos_center_ideal[2]) ** 2 )

points_ideal = points_ideal[points_ideal.distance_from_center < 1.05 * points_dft.distance_from_center.max()]
points_ideal = points_ideal.reset_index(drop=True)

# match the two lattices
def find_closest_points(df1, df2):
    closest_points = []
    for _, row1 in df1.iterrows():
        distances = np.sqrt((df2['x'] - row1['x'])**2 + (df2['y'] - row1['y'])**2 + (df2['z'] - row1['z'])**2)
        closest_index = distances.idxmin()
        closest_points.append(df2.loc[closest_index])
    return pd.DataFrame(closest_points)

closest_points_df = find_closest_points(points_dft, points_ideal)
points_dft[['ideal_x','ideal_y','ideal_z','n1','n2','n3','n4']] = closest_points_df[['x','y','z','n1','n2','n3','n4']].values
points_dft['displacement'] = np.sqrt( (points_dft.x - points_dft.ideal_x)**2 + (points_dft.y - points_dft.ideal_y)**2 + (points_dft.z - points_dft.ideal_z)**2 )

for n in ['n1','n2','n3','n4']:
    points_dft[n] = points_dft[n].astype('int')

points_dft[['z','ideal_z']] -= np.linalg.norm(tau) / 2
points_dft = points_dft.drop(columns=['index','distance','distance_from_center'])

points_dft['distance_from_origin'] = np.sqrt(points_dft.x**2 + points_dft.y**2 + points_dft.z**2)
points_dft = points_dft.sort_values('distance_from_origin')
points_dft = points_dft.reset_index(drop='True')

points_dft = points_dft[['n1','n2','n3','n4','type','x','y','z','ideal_x','ideal_y','ideal_z','displacement','distance_from_origin','Axx','Ayy','Azz','Axy','Axz','Ayz']]

# export the indices-hyperfine tensors to csv
points_dft[['n1','n2','n3','n4','type','Axx','Ayy','Azz','Axy','Axz','Ayz']].iloc[2:].to_csv(output_file)