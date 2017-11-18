#!/usr/bin/env python -i

import dynaphopy.orm.dynamics as dyn
import numpy as np
from dynaphopy.power_spectrum import _progress_bar
from lammps import lammps, PyLammps
from dynaphopy.interface.iofile import get_correct_arrangement

def generate_lammps_trajectory(structure,
                               input_file,
                               total_time=0.1,  # picoseconds
                               time_step=0.002,  # picoseconds
                               relaxation_time=0,
                               silent=False,
                               supercell=(1, 1, 1),
                               memmap=False):

    sampling=1

    lmp = lammps(cmdargs=['-echo','none', '-log', 'none', '-screen', 'none'])

    # test out various library functions after running in.demo

    lmp.file(input_file)
    lmp.command('timestep {}'.format(time_step))
    lmp.command('replicate {} {} {}'.format(*supercell))
    lmp.command('run 0')

    #natoms = lmp.extract_global("natoms",0)
    #mass = lmp.extract_atom("mass",2)

    #print("Natoms, mass, x[0][0] coord =", natoms, mass[1], x[0][0])
    #print ('thermo', lmp.get_thermo('1'))
    temp = lmp.extract_compute("thermo_temp",0,0)

    #print("Temperature from compute =",temp)


    xlo =lmp.extract_global("boxxlo", 1)
    xhi =lmp.extract_global("boxxhi", 1)
    ylo =lmp.extract_global("boxylo", 1)
    yhi =lmp.extract_global("boxyhi", 1)
    zlo =lmp.extract_global("boxzlo", 1)
    zhi =lmp.extract_global("boxzhi", 1)
    xy =lmp.extract_global("xy", 1)
    yz =lmp.extract_global("yz", 1)
    xz =lmp.extract_global("xz", 1)

    simulation_cell = np.array([[xhi-xlo, xy,  xz],
                           [0,  yhi-ylo,  yz],
                           [0,   0,  zhi-zlo]]).T

    positions = []
    velocity = []
    energy = []

    xc = lmp.gather_atoms("x", 1, 3)
    reference = np.array(xc).reshape((-1, 3))
    template = get_correct_arrangement(reference, structure)
    indexing = np.argsort(template)

    lmp.command('run {}'.format(int(relaxation_time/time_step)))

    if not silent:
        _progress_bar(0, 'lammps')

    n_loops = int(total_time/time_step/sampling)
    for i in range(n_loops):

        if not silent:
            _progress_bar(float((i+1) * time_step * sampling) / total_time, 'lammps', )

        lmp.command('run {}'.format(sampling))

        xc = lmp.gather_atoms("x", 1, 3)
        vc = lmp.gather_atoms("v", 1, 3)

        positions.append(np.array(xc).reshape((-1,3))[indexing, :])
        velocity.append(np.array(vc).reshape((-1,3))[indexing, :])
        energy.append(lmp.gather_atoms("pe",1,1))

        #energy.append(lmp.extract_variable("eng",None,0))

    positions = np.array(positions)
    velocity = np.array(velocity)
    energy = np.array(energy)

    lmp.close()

    time = np.array([i * time_step * sampling for i in range(positions.shape[0])], dtype=float)

    return dyn.Dynamics(structure=structure,
                        trajectory=np.array(positions,dtype=complex),
                        #velocity=np.array(velocity,dtype=complex),
                        energy=np.array(energy),
                        time=time,
                        supercell=simulation_cell,
                        memmap=memmap)

if __name__ == '__main__':

    structure = None
    print (generate_lammps_trajectory(structure, 'in.demo'))