import sys
import curses
import numpy as np
import phonopy.file_IO as file_IO
import textwrap
import dynaphopy

from time import sleep
from fractions import Fraction
from os.path import isfile


def list_on_screen(screen, pile, posx, posy):

    pile = np.array(pile).reshape((-1,3))

    for i, row_list in enumerate(pile):
        for j, element in enumerate(row_list):
            screen.addstr(posx+i,posy+j*20, str(i*len(pile[0])+j+1)+": {0:.4f}".format(element))


# Get parametres from text ui
def get_param(screen, prompt_string):

    screen_width = 80
    prompt_string = textwrap.fill(prompt_string, width=screen_width)

    screen.clear()
    screen.border(0)

    for i, line in enumerate(prompt_string.splitlines()):
        screen.addstr(2+i, 2, line)
    screen.refresh()
    input_data = screen.getstr(10, 10, 60).decode()

    return input_data


def interactive_interface(calculation, trajectory, args, structure_file):

    screen = curses.initscr()
    screen.border(0)
    screen.addstr(5, 7, "Welcome   to   DynaPhoPy " + dynaphopy.__version__)
    screen.refresh()
    sleep(3)
    curses.endwin()

    x = 0
    while x != ord('0'):
        screen = curses.initscr()
        screen.clear()
        screen.border(0)

        #Show parameters right screen
        screen.addstr(2,45,"Input file: " + args.input_file[0][-20:])
        if args.load_data:
            screen.addstr(4,45,"hdf5 file: " + args.load_data[0][-20:])
        else:
            screen.addstr(3,45,"Structure file: " + structure_file[-14:])
            if args.md_file:
                screen.addstr(4,45,"MD file: " + args.md_file[-20:])
            else:
                screen.addstr(4,45, "Generated trajectory")

        screen.addstr(6,45,"Wave vector: {0} ".format(calculation.get_reduced_q_vector()))
        screen.addstr(7,45,"Frequency range: {0} - {1} THz".format(calculation.get_frequency_range()[0],
                                                                   calculation.get_frequency_range()[-1]))
        screen.addstr(8,45,"Pow. spectr. resolution: {0} THz".format(calculation.parameters.spectrum_resolution))

        screen.addstr(10,45,"Primitive cell atoms: {0}".format(trajectory.structure.get_number_of_primitive_atoms()))
        screen.addstr(11,45,"Unit cell atoms: {0}".format(trajectory.structure.get_number_of_atoms()))
        screen.addstr(12,45,"MD supercell atoms: {0} ".format(trajectory.get_number_of_atoms()))
        screen.addstr(13,45,"Number of MD time steps: {0}".format(len(trajectory.velocity)))
        screen.addstr(14,45,"Time step: {0} ps".format(np.round(trajectory.get_time_step_average(),decimals=12)))


        #Option values left screen
        screen.addstr(2, 2, "Please enter option number...")
        screen.addstr(4, 4, "1 - Harmonic calculations")
        screen.addstr(5, 4, "2 - Change wave vector")
        screen.addstr(6, 4, "3 - Thermal properties")
        screen.addstr(7, 4, "4 - Maxwell-Boltzmann analysis")
        screen.addstr(8, 4, "5 - Power spectrum")
        screen.addstr(9, 4, "6 - Renormalized phonon dispersion")
        screen.addstr(10, 4, "7 - Peak analysis")
        screen.addstr(11, 4, "8 - Atomic displacements distribution")
        screen.addstr(12, 4, "9 - Preferences")
        screen.addstr(14, 4, "0 - Exit")

        screen.refresh()

        x = screen.getch()

######## OPTION 1 :  DISPLAY HARMONIC DATA
        if x == ord('1'):
            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Display...")
                screen.addstr(4, 4, "1 - Show harmonic frequencies")
                screen.addstr(5, 4, "2 - Show harmonic eigenvectors")
                screen.addstr(6, 4, "3 - Plot phonon dispersion relations")
                screen.addstr(7, 4, "4 - Plot phonon density of states")
                screen.addstr(9, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()

                curses.endwin()

                if x2 == ord('1'):
                    freq = calculation.get_frequencies()
                    sleep(1)
                    screen = curses.initscr()
                    screen.clear()
                    screen.border()

                    screen.addstr(2, 4, "Frequencies (THz)")
                    screen.addstr(3, 4, "-----------------")

                    list_on_screen(screen, freq, 5, 4)
                    screen.getch()

                if x2 == ord('2'):
                    calculation.plot_eigenvectors()

                if x2 == ord('3'):
                    calculation.plot_phonon_dispersion_bands()

                if x2 == ord('4'):
                    calculation.plot_dos_phonopy()

######## OPTION 2 :  DEFINE WAVE VECTOR
        if x == ord('2'):
            q_vector = np.array([float(Fraction(s)) for s in
                                 get_param(screen, "Insert reduced wave vector (values separated by comma)").split(',')])
            calculation.set_reduced_q_vector(q_vector)

######## OPTION 3 :  THERMAL PROPERTIES
        if x == ord('3'):
            curses.endwin()
            calculation.display_thermal_properties()

######## OPTION 4 :  BOLTZMANN DISTRIBUTION
        if x == ord('4'):
            curses.endwin()
            calculation.show_boltzmann_distribution()

######## OPTION 5 :  PLOTTING POWER SPECTRA
        if x == ord('5'):

            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Plotting...")
                screen.addstr(4, 4, "1 - Full power spectrum")
                screen.addstr(5, 4, "2 - Wave vector projection power spectrum")
                screen.addstr(6, 4, "3 - Phonon mode projection power spectrum")
                screen.addstr(8, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()
                curses.endwin()

                if x2 == ord('1'):
                    calculation.plot_power_spectrum_full()

                if x2 == ord('2'):
                    calculation.plot_power_spectrum_wave_vector()

                if x2 == ord('3'):
                    calculation.plot_power_spectrum_phonon()

######## OPTION 6 :  Renormalized phonon dispersion
        if x == ord('6'):
            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Plotting...")
                screen.addstr(4, 4, "1 - Harmonic and renormalized phonon dispersion relations")
                screen.addstr(5, 4, "2 - Renormalized phonon dispersions and linewidths (fat bands)")
                screen.addstr(6, 4, "3 - Frequency shifts and linewidths (separated)")
                screen.addstr(7, 4, "4 - Frequency vs linewidth (interpolated mesh)")
                screen.addstr(9, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()

                curses.endwin()

                if x2 == ord('1'):
                    calculation.plot_renormalized_phonon_dispersion_bands()

                if x2 == ord('2'):
                    calculation.plot_renormalized_phonon_dispersion_bands(plot_linewidths=True,
                                                                          plot_harmonic=False)

                if x2 == ord('3'):
                    calculation.plot_linewidths_and_shifts_bands()

                if x2 == ord('4'):
                    calculation.plot_frequencies_vs_linewidths()


######## OPTION 7 :  PEAK ANALYSIS
        if x == ord('7'):
            curses.endwin()
            calculation.phonon_individual_analysis()
            screen.getch()

######## OPTION 8 :  TRAJECTORY DISTRIBUTION
        if x == ord('8'):
            direction = np.array([float(Fraction(s)) for s in
                                 get_param(screen, "Introduce the vector that defines direction in real space (x,y,z) "
                                                   "in which the distribution will be calculated (values separated by comma)").split(',')])
            curses.endwin()
            calculation.plot_trajectory_distribution(direction)

        ######## OPTION 9 :  PREFERENCES
        if x == ord('9'):

            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Preferences...")
                screen.addstr(4, 4, "1 - Select power spectrum algorithm")
                screen.addstr(5, 4, "2 - Non analytical corrections (dispersion spectrum only): {0}".format(
                        calculation.parameters.use_NAC))
                screen.addstr(6, 4, "3 - Number of MEM coefficients: {0}".format(
                        calculation.parameters.number_of_coefficients_mem))
                screen.addstr(7, 4, "4 - Number of bins in histograms (Boltzmann/displacements): {0}".format(
                        calculation.parameters.number_of_bins_histogram))
                screen.addstr(8, 4, "5 - Eigenvectors display vector scale: {0}".format(
                        calculation.parameters.modes_vectors_scale))
                screen.addstr(9, 4, "6 - Select fitting function")
                screen.addstr(10, 4, "7 - Change spectrum resolution")
                screen.addstr(11, 4, "8 - Change frequency range")

                screen.addstr(13, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()
                # SUB OPT 1 : # PS select algorithm
                if x2 == ord('1'):
                    x3 = 9
                    while x3 >= len(calculation.get_algorithm_list()):
                        screen = curses.initscr()
                        screen.clear()
                        screen.border(0)

                        screen.addstr(2, 2, "Algorithms...")
                        for i, algorithm in enumerate(calculation.get_algorithm_list()):
                            if i == calculation.parameters.power_spectra_algorithm:
                                screen.addstr(4+i, 3, ">"+str(i) +" : "+ str(algorithm[1]))
                            else:
                                screen.addstr(4+i, 4, str(i) +" : "+ str(algorithm[1]))

                        screen.refresh()
                        try:
                            x3 = int(chr(int(screen.getch())))
                        except ValueError:
                            x3 = 9

                    calculation.select_power_spectra_algorithm(x3)

                    curses.endwin()

                # SUB OPT 2 : # Non analytical corrections
                if x2 == ord('2'):
                    x3 = ord("9")
                    while int(chr(int(x3))) > 2:
                        screen = curses.initscr()
                        screen.clear()
                        screen.border(0)

                        screen.addstr(2, 2, "Non analytical corrections...")
                        if isfile("BORN"):
                            screen.addstr(4, 4, "1 - On  (BORN file found)")
                        else:
                            screen.addstr(4, 4, "1 - On  (Warning: BORN file not found)")
                        screen.addstr(5, 4, "2 - Off")


                        if calculation.parameters.use_NAC:
                            screen.addstr(4, 3, ">")
                        else:
                            screen.addstr(5, 3, ">")


                        screen.refresh()
                        x3 = screen.getch()
                    if isfile("BORN"):
                        calculation.set_NAC(bool(int(chr(int(x3)))-2))

                    curses.endwin()

                # SUB OPT 3 : # Number of MEM coefficients
                if x2 == ord('3'):
               # calculation.set_number_of_mem_coefficients(int(get_param(screen, "Insert number of coefficients")))
                    calculation.parameters.number_of_coefficients_mem = int(get_param(screen, "Insert number of coefficients"))
                    curses.endwin()

                # SUB OPT 4 : # Number of bins histogram (coordinates and Boltzmann dist.)
                if x2 == ord('4'):
                    calculation.parameters.number_of_bins_histogram = int(get_param(screen, "Insert number of bins"))
                    curses.endwin()

                # SUB OPT 5 : # Vector scale (phonon modes)
                if x2 == ord('5'):
                    calculation.parameters.modes_vectors_scale = int(get_param(screen, "Insert vector scale"))
                    curses.endwin()

                # SUB OPT 6 : # Fitting function selection
                if x2 == ord('6'):
                    from dynaphopy.analysis.fitting.fitting_functions import fitting_functions
                    x3 = 9
                    while x3 >= len(fitting_functions):
                        screen = curses.initscr()
                        screen.clear()
                        screen.border(0)

                        screen.addstr(2, 2, "Fitting functions...")
                        for i in fitting_functions.keys():
                            algorithm = fitting_functions[i]
                            if i == calculation.parameters.fitting_function:
                                screen.addstr(4+i, 3, ">"+str(i) +" : "+ str(algorithm).split('.')[-1].replace('_',' '))
                            else:
                                screen.addstr(4+i, 4, str(i) +" : "+ str(algorithm).split('.')[-1].replace('_',' '))

                        screen.refresh()
                        try:
                            x3 = int(chr(int(screen.getch())))
                        except ValueError:
                            x3 = 9

                    calculation.select_fitting_function(x3)

                    curses.endwin()

                # SUB OPT 7 : # Resolution
                if x2 == ord('7'):
                    resolution =float(get_param(screen, "Insert resolution in THz"))
                    calculation.set_spectra_resolution(resolution)
                    calculation.full_clear()
                    curses.endwin()

                # SUB OPT 8 : # Frequency range
                if x2 == ord('8'):
                    frequency_limits = np.array([float(Fraction(s)) for s in
                                                 get_param(screen, "Insert frequency range (min, max)").split(',')])
                    print(frequency_limits)
                    calculation.set_frequency_limits(frequency_limits)
                    curses.endwin()

    curses.endwin()

# Testing
if __name__ == 'test_gui.py':
    import dynaphopy as controller
    import dynaphopy.functions.iofunctions as reading

    # Get data from input file
    input_parameters = reading.read_parameters_from_input_file(sys.argv[1])

    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name'])
    structure.set_force_set( file_IO.parse_FORCE_SETS(filename=input_parameters['force_constants_file_name']))
    structure.set_primitive_matrix(input_parameters['primitive_matrix'])
    structure.set_supercell_phonon(input_parameters['supercell_matrix'])

    trajectory_file_name = sys.argv[2]

    trajectory = reading.read_from_file_trajectory(trajectory_file_name,structure,last_steps=5000)

    calculation = controller.Quasiparticle(trajectory)
    calculation.set_band_ranges(input_parameters['bands'])