import dynaphopy.dynamics as dyn
import numpy as np
from dynaphopy.power_spectrum import _progress_bar
from lammps import lammps
from dynaphopy.interface.iofile import get_correct_arrangement


def generate_lammps_trajectory(structure,
                               input_file,
                               total_time=0.1,  # picoseconds
                               time_step=0.002,  # picoseconds
                               relaxation_time=0,
                               silent=False,
                               supercell=(1, 1, 1),
                               memmap=False,  # not fully implemented yet!
                               velocity_only=False,
                               lammps_log=True,
                               temperature=None,
                               thermostat_mass=0.5,
                               sampling_interval=1):  # in timesteps

    cmdargs_lammps = ['-echo','none', '-screen', 'none']
    if not lammps_log:
        cmdargs_lammps += ['-log', 'none']

    lmp = lammps(cmdargs=cmdargs_lammps)

    lmp.file(input_file)
    lmp.command('timestep {}'.format(time_step))

    lmp.command('replicate {} {} {}'.format(*supercell))

    # Force reset temperature (overwrites lammps script)
    # This forces NVT simulation
    if temperature is not None:
        print ('Temperature reset to {} (NVT)'.format(temperature))
        lmp.command('fix             int all nvt temp {0} {0} {1}'.format(temperature,
                                                                          thermostat_mass))
    # Check if correct number of atoms
    if lmp.extract_global("natoms",0) < 2:
        print ('Number of atoms in MD should be higher than 1!')
        exit()

    # Check if initial velocities all zero
    if not np.array(lmp.gather_atoms("v", 1, 3)).any():
        print('Set lammps initial velocities')
        t = temperature if temperature is not None else 100
        lmp.command('velocity        all create {} 3627941 dist gaussian mom yes'.format(t))
        lmp.command('velocity        all scale {}'.format(t))

    lmp.command('run 0')

    # natoms = lmp.extract_global("natoms",0)
    # mass = lmp.extract_atom("mass",2)
    # temp = lmp.extract_compute("thermo_temp",0,0)
    # print("Temperature from compute =",temp)
    # print("Natoms, mass, x[0][0] coord =", natoms, mass[1], x[0][0])
    # print ('thermo', lmp.get_thermo('1'))

    try:
        xlo =lmp.extract_global("boxxlo")
        xhi =lmp.extract_global("boxxhi")
        ylo =lmp.extract_global("boxylo")
        yhi =lmp.extract_global("boxyhi")
        zlo =lmp.extract_global("boxzlo")
        zhi =lmp.extract_global("boxzhi")
        xy =lmp.extract_global("xy")
        yz =lmp.extract_global("yz")
        xz =lmp.extract_global("xz")

    except TypeError:
        xlo =lmp.extract_global("boxxlo", 1)
        xhi =lmp.extract_global("boxxhi", 1)
        ylo =lmp.extract_global("boxylo", 1)
        yhi =lmp.extract_global("boxyhi", 1)
        zlo =lmp.extract_global("boxzlo", 1)
        zhi =lmp.extract_global("boxzhi", 1)
        xy =lmp.extract_global("xy", 1)
        yz =lmp.extract_global("yz", 1)
        xz =lmp.extract_global("xz", 1)

    except UnboundLocalError:
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        xlo, ylo, zlo = boxlo
        xhi, yhi, zhi = boxhi

    simulation_cell = np.array([[xhi-xlo, xy,  xz],
                           [0,  yhi-ylo,  yz],
                           [0,   0,  zhi-zlo]]).T

    positions = []
    velocity = []
    energy = []

    na = lmp.get_natoms()
    id = lmp.extract_atom("id", 0)
    id = np.array([id[i] - 1 for i in range(na)], dtype=int)

    xp = lmp.extract_atom("x", 3)
    reference = np.array([[xp[i][0], xp[i][1], xp[i][2]] for i in range(na)], dtype=float)[id, :]

    # xc = lmp.gather_atoms("x", 1, 3)
    # reference = np.array([xc[i] for i in range(na*3)]).reshape((na,3))

    template = get_correct_arrangement(reference, structure)
    indexing = np.argsort(template)

    lmp.command('variable energy equal pe'.format(int(relaxation_time/time_step)))
    lmp.command('run {}'.format(int(relaxation_time/time_step)))

    if not silent:
        _progress_bar(0, 'lammps')

    n_loops = int(total_time / time_step / sampling_interval)
    for i in range(n_loops):
        if not silent:
            _progress_bar(float((i+1) * time_step * sampling_interval) / total_time, 'lammps', )

        lmp.command('run {}'.format(sampling_interval))

        # xc = lmp.gather_atoms("x", 1, 3)
        # vc = lmp.gather_atoms("v", 1, 3)
        energy.append(lmp.extract_variable('energy', 'all', 0))

        id = lmp.extract_atom("id", 0)
        id = np.array([id[i] - 1 for i in range(na)], dtype=int)

        vc = lmp.extract_atom("v", 3)
        velocity.append(np.array([[vc[i][0], vc[i][1], vc[i][2]] for i in range(na)], dtype=float)[id, :][indexing, :])
        # velocity.append(np.array([vc[i] for i in range(na * 3)]).reshape((na, 3))[indexing, :])

        if not velocity_only:
            xc = lmp.extract_atom("x", 3)
            positions.append(np.array([[xc[i][0], xc[i][1], xc[i][2]] for i in range(na)], dtype=float)[id, :][indexing, :])
            # positions.append(np.array([xc[i] for i in range(na * 3)]).reshape((na, 3))[indexing, :])

    positions = np.array(positions, dtype=complex)
    velocity = np.array(velocity, dtype=complex)
    energy = np.array(energy)

    if velocity_only:
        positions = None

    lmp.close()

    time = np.array([i * time_step * sampling_interval for i in range(n_loops)], dtype=float)


    return dyn.Dynamics(structure=structure,
                        trajectory=positions,
                        velocity=velocity,
                        time=time,
                        energy=energy,
                        supercell=simulation_cell,
                        memmap=memmap)


if __name__ == '__main__':

    structure = None
    print (generate_lammps_trajectory(structure, 'in.demo'))