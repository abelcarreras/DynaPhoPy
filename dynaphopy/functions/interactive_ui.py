from time import sleep
import sys
import curses
import numpy as np

import phonopy.file_IO as file_IO



#Display list on screen
from dynaphopy import functions as reading


def list_on_screen(screen,pile,posx,posy):

    pile = np.array(pile).reshape((-1,3))

    for i, row_list in enumerate(pile):
        for j, element in enumerate(row_list):
            screen.addstr(posx+i,posy+j*20,str(i*len(pile[0])+j+1)+": "+str(element)[:8])

# Get parametres from text ui
def get_param(screen,prompt_string):
    screen.clear()
    screen.border(0)
    screen.addstr(2, 2, prompt_string)
    screen.refresh()
    input_data = screen.getstr(10, 10, 60)
    return input_data

def interactive_interface(calculation,trajectory, args, structure_file):

    screen = curses.initscr()
    screen.border(0)
    screen.addstr(5, 7, "Welcome   to   DynaPhoPy 1.1")
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
        screen.addstr(3,45,"Structure file: "+ structure_file[-20:])
        screen.addstr(4,45,"MD file: "+ args.md_file[0][-20:])

        screen.addstr(6,45,"Wave Vector: "+str(calculation.get_reduced_q_vector()))
        screen.addstr(7,45,"Frequency range: "+str(calculation.get_frequency_range()[0])+' - '
                                              +str(calculation.get_frequency_range()[-1])+' THz')
        screen.addstr(8,45,"Primitive cell atoms: "+str(trajectory.structure.get_number_of_primitive_atoms()))
        screen.addstr(9,45,"Unit cell atoms: "+str(trajectory.structure.get_number_of_atoms()))
        screen.addstr(10,45,"MD  cell atoms: "+str(trajectory.get_number_of_atoms()))
        screen.addstr(11,45,"Number of MD steps: "+str(len(trajectory.get_time())))


        #Option values left screen
        screen.addstr(2, 2, "Please enter an option number...")
        screen.addstr(4, 4, "1 - Harmonic data")
        screen.addstr(5, 4, "2 - Change wave vector")
        screen.addstr(6, 4, "3 - Change frequency range")
        screen.addstr(7, 4, "4 - Boltzmann analysis")
        screen.addstr(8, 4, "5 - Plot power spectrum")
        screen.addstr(9, 4, "6 - Save power spectrum")
        screen.addstr(11, 4, "0 - Exit")

        screen.refresh()

        x = screen.getch()

        if x == ord('1'):
            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Display...")
                screen.addstr(4, 4, "1 - Show harmonic frequencies")
                screen.addstr(5, 4, "2 - Show harmonic eigenvectors")
                screen.addstr(6, 4, "3 - Plot phonon dispersion spectra")
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

                    screen.addstr(2,4,"Frequencies")
                    screen.addstr(3,4,"------------")

                    list_on_screen(screen,freq,5,4)

      #              for i,freq in enumerate(freq):
      #                  screen.addstr(4+i,4,str(i)+": "+str(freq))
                    screen.getch()


                if x2 == ord('2'):
                    curses.endwin()
                    calculation.get_eigenvectors()

                if x2 == ord('3'):
                    curses.endwin()
                    calculation.get_phonon_dispersion_spectra()

        if x == ord('2'):
            q_vector = np.array(get_param(screen,"Insert reduced wave vector (values separated by comma)").split(','),dtype=float)
            calculation.set_reduced_q_vector(q_vector)
            curses.endwin()

        if x == ord('3'):
            frequency_limits = np.array(get_param(screen,"Insert frequency range (min, max, number of points)").split(','),dtype=float)
            print(frequency_limits)
            calculation.set_frequency_range(np.linspace(*frequency_limits))
            curses.endwin()

        if x == ord('4'):
            curses.endwin()
            calculation.show_boltzmann_distribution()
            curses.endwin()


        if x == ord('5'):

            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Plotting...")
                screen.addstr(4, 4, "1 - Real space power spectrum")
                screen.addstr(5, 4, "2 - Wave vector projection power spectrum")
                screen.addstr(6, 4, "3 - Phonon mode projection power spectrum")
                screen.addstr(8, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()


                if x2 == ord('1'):
                    curses.endwin()
                    calculation.plot_correlation_direct()

                if x2 == ord('2'):
                    curses.endwin()
                    calculation.plot_correlation_wave_vector()

                if x2 == ord('3'):
                    curses.endwin()
                    calculation.plot_correlation_phonon()

        if x == ord('6'):

            x2 = 0
            while x2 != ord('0'):
                screen = curses.initscr()
                screen.clear()
                screen.border(0)

                screen.addstr(2, 2, "Saving...")
                screen.addstr(4, 4, "1 - Real space power spectrum")
                screen.addstr(5, 4, "2 - Wave vector projection power spectrum")
                screen.addstr(6, 4, "3 - Phonon mode projection power spectrum")
                screen.addstr(8, 4, "0 - Return")
                screen.refresh()

                x2 = screen.getch()

                if x2 == ord('1'):
                    save_file = get_param(screen,'Insert file name')
                    curses.endwin()
                    calculation.write_correlation_direct(save_file)

                if x2 == ord('2'):
                    save_file = get_param(screen,'Insert file name')
                    curses.endwin()
                    calculation.write_correlation_wave_vector(save_file)

                if x2 == ord('3'):
                    save_file = get_param(screen,'Insert file name')
                    curses.endwin()
                    calculation.write_correlation_phonon(save_file)

    curses.endwin()

#Just for testing
if __name__ == 'test_gui.py':
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