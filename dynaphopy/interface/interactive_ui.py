from time import sleep
import sys
import curses
import numpy as np
import phonopy.file_IO as file_IO
from fractions import Fraction
from os.path import isfile
import textwrap

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

    for i,line in enumerate(prompt_string.splitlines()):
        screen.addstr(2+i, 2, line)
    screen.refresh()
    input_data = screen.getstr(10, 10, 60)

    return input_data


def interactive_interface(calculation, trajectory, args, structure_file):

    screen = curses.initscr()
    screen.border(0)
    screen.addstr(5, 7, "Welcome   to   DynaPhoPy 1.5")
    screen.refresh()
    sleep(3)
    curses.endwin()

    x = 0
    while x != ord('0'):
        screen = curses.initscr()
        screen.clear()
        screen.border(0)

        #Show parameters right screen
        screen.addstr(2,45,"Input file: "+args.input_file[0][-20:])
        if args.load_velocity:
            screen.addstr(4,45,"hdf5 file: "+ args.load_velocity[0][-20:])
        else:
            screen.addstr(3,45,"Structure file: "+ structure_file[-14:])
            screen.addstr(4,45,"MD file: "+ args.md_file[-20:])

        screen.addstr(6,45,"Wave vector: "+str(calculation.get_reduced_q_vector()))
        screen.addstr(7,45,"Frequency range: "+str(calculation.get_frequency_range()[0])+' - '
                                              +str(calculation.get_frequency_range()[-1])+' THz')
        screen.addstr(9,45,"Primitive cell atoms: "+str(trajectory.structure.get_number_of_primitive_atoms()))
        screen.addstr(10,45,"Unit cell atoms: "+str(trajectory.structure.get_number_of_atoms()))
        screen.addstr(11,45,"MD  cell atoms: "+str(trajectory.get_number_of_atoms()))
        screen.addstr(12,45,"Number of MD time steps: "+str(len(trajectory.velocity)))


        #Option values left screen
        screen.addstr(2, 2, "Please enter option number...")
        screen.addstr(4, 4, "1 - Harmonic data")
        screen.addstr(5, 4, "2 - Change wave vector")
        screen.addstr(6, 4, "3 - Change frequency range")
        screen.addstr(7, 4, "4 - Boltzmann analysis")
        screen.addstr(8, 4, "5 - Power spectrum")
        screen.addstr(9, 4, "6 - Renormalized phonon dispersion")
        screen.addstr(10, 4, "7 - Peak analysis")
        screen.addstr(11, 4, "8 - Atomic displacements")
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
                screen.addstr(6, 4, "3 - Plot phonon dispersion bands")
                screen.addstr(8, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()


                if x2 == ord('1'):
                    curses.endwin()
                    freq = calculation.get_frequencies()
                    sleep(1)
                    screen = curses.initscr()
                    screen.clear()
                    screen.border()

                    screen.addstr(2,4,"Frequencies (THz)")
                    screen.addstr(3,4,"-----------------")

                    list_on_screen(screen,freq,5,4)

      #              for i,freq in enumerate(freq):
      #                  screen.addstr(4+i,4,str(i)+": "+str(freq))
                    screen.getch()


                if x2 == ord('2'):
                    curses.endwin()
                    #calculation.get_eigenvectors()
                    calculation.plot_eigenvectors()

                if x2 == ord('3'):
                    curses.endwin()
                    calculation.get_phonon_dispersion_spectra()

######## OPTION 2 :  DEFINE WAVE VECTOR
        if x == ord('2'):
            q_vector = np.array([float(Fraction(s)) for s in
                                 get_param(screen, "Insert reduced wave vector (values separated by comma)").split(',')])
            calculation.set_reduced_q_vector(q_vector)
            curses.endwin()

######## OPTION 3 :  DEFINE FREQUENCY RANGE
        if x == ord('3'):
            frequency_limits = np.array([float(Fraction(s)) for s in
                                         get_param(screen, "Insert frequency range (min, max, number of points)").split(',')])
            print(frequency_limits)
            calculation.set_frequency_range(np.linspace(*frequency_limits))
            curses.endwin()

######## OPTION 4 :  BOLTZMANN DISTRIBUTION
        if x == ord('4'):
            curses.endwin()
            calculation.show_boltzmann_distribution()
            curses.endwin()

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


                if x2 == ord('1'):
                    curses.endwin()
                    calculation.plot_power_spectrum_full()

                if x2 == ord('2'):
                    curses.endwin()
                    calculation.plot_power_spectrum_wave_vector()

                if x2 == ord('3'):
                    curses.endwin()
                    calculation.plot_power_spectrum_phonon()

######## OPTION 6 :  Renormalized phonon dispersion
        if x == ord('6'):
            curses.endwin()
            calculation.get_renormalized_phonon_dispersion_spectra()
            curses.endwin()

######## OPTION 7 :  PEAK ANALYSIS
        if x == ord('7'):
            curses.endwin()
            calculation.phonon_individual_analysis()
            screen.getch()
            curses.endwin()

######## OPTION 8 :  TRAJECTORY DISTRIBUTION
        if x == ord('8'):
            direction = np.array([float(Fraction(s)) for s in
                                 get_param(screen, "Insert the vector that defines direction in which the "
                                                   "distribution will be calculated (values separated by comma)").split(',')])
            curses.endwin()
            calculation.plot_trajectory_distribution(direction)


######## OPTION 9 :  PREFERENCES  (UNDER DEVELOPMENT)
        if x == ord('9'):

            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Preferences...")
                screen.addstr(4, 4, "1 - Power spectrum algorithm")
                screen.addstr(5, 4, "2 - Non analytical corrections (dispersion spectrum only): " +
                            str(calculation.parameters.use_NAC))
                screen.addstr(6, 4, "3 - Number of MEM coefficients: " +
                              str(calculation.parameters.number_of_coefficients_mem))
                screen.addstr(7, 4, "4 - Number of bins in histograms (Boltzman/displacements): " +
                              str(calculation.parameters.number_of_bins_histogram))
                screen.addstr(8, 4, "5 - Eigenvectors display vector scale: " +
                              str(calculation.parameters.modes_vectors_scale))

                screen.addstr(10, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()

                if x2 == ord('1'):
                    x3 = 9
                    while x3 >= len(calculation.get_algorithm_list()):
                        screen = curses.initscr()
                        screen.clear()
                        screen.border(0)

                        screen.addstr(2, 2, "Algorithms...")
                        for i, algorithm in enumerate(calculation.get_algorithm_list()):
                            if i == calculation.parameters.power_spectra_algorithm:
                                screen.addstr(4+i, 3, ">"+str(i) +" : "+ str(algorithm))
                            else:
                                screen.addstr(4+i, 4, str(i) +" : "+ str(algorithm))

                        screen.refresh()
                        try:
                            x3 = int(chr(int(screen.getch())))
                        except ValueError:
                            x3 = 9

                    calculation.select_power_spectra_algorithm(x3)

                    curses.endwin()

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

                if x2 == ord('3'):
               #     calculation.set_number_of_mem_coefficients(int(get_param(screen, "Insert number of coefficients")))
                    calculation.parameters.number_of_coefficients_mem = int(get_param(screen, "Insert number of coefficients"))

                    curses.endwin()

                if x2 == ord('4'):
                    calculation.parameters.number_of_bins_histogram = int(get_param(screen, "Insert number of bins"))
                    curses.endwin()

                if x2 == ord('5'):
                    calculation.parameters.modes_vectors_scale = int(get_param(screen, "Insert vector scale"))
                    curses.endwin()


    curses.endwin()



#Just for testing
if __name__ == 'test_gui.py':
    import dynaphopy as controller
    import dynaphopy.functions.iofunctions as reading
    #Get data from input file
    input_parameters = reading.read_parameters_from_input_file(sys.argv[1])

    structure = reading.read_from_file_structure_outcar(input_parameters['structure_file_name'])
    structure.set_force_set( file_IO.parse_FORCE_SETS(filename=input_parameters['force_constants_file_name']))
    structure.set_primitive_matrix(input_parameters['primitive_matrix'])
    structure.set_super_cell_phonon(input_parameters['super_cell_matrix'])

    trajectory_file_name = sys.argv[2]

    trajectory = reading.read_from_file_trajectory(trajectory_file_name,structure,last_steps=5000)

    calculation = controller.Calculation(trajectory)
    calculation.set_band_ranges(input_parameters['bands'])