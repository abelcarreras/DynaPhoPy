#!/usr/bin/env python

import h5py
import sys
import numpy as np

def read_file_hdf5(file_name):

    hdf5_file = h5py.File(file_name, "r")
    velocity = hdf5_file['velocity'][:]
    time = hdf5_file['time'][:]
    super_cell = hdf5_file['super_cell'][:]
    hdf5_file.close()

    return velocity, time, super_cell

def save_data_hdf5(file_name, velocity, time, super_cell):
    hdf5_file = h5py.File(file_name, "w")

    hdf5_file.create_dataset('velocity', data=velocity)
    hdf5_file.create_dataset('time', data=time)
    hdf5_file.create_dataset('super_cell', data=super_cell)

    print "saved", velocity.shape[0], "steps"
    hdf5_file.close()

last = 50000
velocity, time, super_cell = read_file_hdf5(sys.argv[1])
velocity = velocity[-last:]
print(sys.argv[1])

for arg in sys.argv[2:-1]:
    print(arg)
    new_velocity = read_file_hdf5(arg)[0]
    velocity = np.concatenate((velocity, new_velocity[-last:]), axis=0)
print "Final: ", sys.argv[-1]

save_data_hdf5(sys.argv[-1], velocity, time, super_cell)

