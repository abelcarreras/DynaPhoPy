#!/usr/bin/env python

from os import system
from time import sleep

import sys
import curses
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller
import numpy as np

#Display list on screen
def list_on_screen(pile,posx,posy):

    pile = np.array(pile).reshape((-1,3))

    for i, row_list in enumerate(pile):
        for j, element in enumerate(row_list):
            screen.addstr(posx+i,posy+j*20,str(i*len(pile[0])+j)+": "+str(element)[:8])


# Get parametres from text ui
def get_param(prompt_string):
    screen.clear()
    screen.border(0)
    screen.addstr(2, 2, prompt_string)
    screen.refresh()
    input_data = screen.getstr(10, 10, 60)
    return input_data


#Get data from input file
input_parameters = reading.read_parameters_from_input_file(sys.argv[1])

structure = reading.read_from_file_structure(input_parameters['structure_file_name'])
structure.set_force_set( file_IO.parse_FORCE_SETS(filename=input_parameters['force_constants_file_name']))
structure.set_primitive_matrix(input_parameters['primitive_matrix'])
structure.set_super_cell_phonon(input_parameters['super_cell_matrix'])

trajectory_file_name = sys.argv[2]

trajectory = reading.read_from_file_trajectory(trajectory_file_name,structure,last_steps=5000)

calculation = controller.Calculation(trajectory)
calculation.set_band_ranges(input_parameters['bands'])


########## SET NAC IF NECESSARY (BANDS PLOT ONLY) ###########
#calculation.set_NAC(True)
calculation.set_frequency_range(np.array([0.1*i + 0.1 for i in range (500)]))
#############################################################

screen = curses.initscr()
screen.border(0)
screen.addstr(5, 7, "Welcome   to   DynaPhoPy 1.0")
screen.refresh()
sleep(3)
curses.endwin()

x = 0
while x != ord('0'):
    screen = curses.initscr()
    screen.clear()
    screen.border(0)

    #Show parameters right screen
    screen.addstr(2,50,"Wave Vector: "+str(calculation.get_reduced_q_vector()))
    screen.addstr(3,50,"Structure: "+input_parameters['structure_file_name'])
    screen.addstr(4,50,"Input file: "+ sys.argv[1])
    screen.addstr(5,50,"Unit cell: ")
    screen.addstr(6,50,"Expansion: ")

    #Option values left screen
    screen.addstr(2, 2, "Please enter an option number...")
    screen.addstr(4, 4, "1 - Eigenvectors/eigenvalues")
    screen.addstr(5, 4, "2 - Change wave vector")
    screen.addstr(6, 4, "3 - Boltzmann analysis")
    screen.addstr(7, 4, "4 - Plot correlation function")
    screen.addstr(8, 4, "5 - Save correlation function")
    screen.addstr(10, 4, "0 - Exit")

    screen.refresh()

    x = screen.getch()

    if x == ord('1'):
        x2 = 0
        while x2 != ord('0'):
            screen = curses.initscr()
            screen.clear()
            screen.border(0)


            screen.addstr(2, 2, "Display...")
            screen.addstr(4, 4, "1 - Frequencies")
            screen.addstr(5, 4, "2 - Eigenvectors")
            screen.addstr(6, 4, "3 - Phonon dispersion spectra")
            screen.addstr(8, 4, "0 - Return")
            screen.refresh()

            x2 = screen.getch()


            if x2 == ord('1'):
                curses.endwin()
                freq = calculation.get_frequencies()
                screen = curses.initscr()
                screen.clear()
                screen.border()

                screen.addstr(2,4,"Frequencies")
                screen.addstr(3,4,"------------")

                list_on_screen(freq,5,4)

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
        q_vector = np.array(get_param("Introduce wave vector (values separated by comma)").split(','),dtype=float)
        calculation.set_reduced_q_vector(q_vector)
        curses.endwin()
        print(q_vector)

    if x == ord('3'):
        curses.endwin()
        calculation.show_boltzmann_distribution()


    if x == ord('4'):

        x2 = 0
        while x2 != ord('0'):
            screen = curses.initscr()
            screen.clear()
            screen.border(0)

            screen.addstr(2, 2, "Plotting...")
            screen.addstr(4, 4, "1 - Real space correlation")
            screen.addstr(5, 4, "2 - Wave vector projection correlation")
            screen.addstr(6, 4, "3 - Phonon mode projection correlation")
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

    if x == ord('5'):

        x2 = 0
        while x2 != ord('0'):
            screen = curses.initscr()
            screen.clear()
            screen.border(0)

            screen.addstr(2, 2, "Saving...")
            screen.addstr(4, 4, "1 - Real space correlation")
            screen.addstr(5, 4, "2 - Wave vector projection correlation")
            screen.addstr(6, 4, "3 - Phonon mode projection correlation")
            screen.addstr(8, 4, "0 - Return")
            screen.refresh()

            x2 = screen.getch()

            if x2 == ord('1'):
                save_file = get_param('Insert file name')
                curses.endwin()
                calculation.write_correlation_direct(save_file)

            if x2 == ord('2'):
                save_file = get_param('Insert file name')
                curses.endwin()
                calculation.write_correlation_wave_vector(save_file)

            if x2 == ord('3'):
                curses.endwin()
                save_file = get_param('Insert file name')
                curses.endwin()
                calculation.write_correlation_phonon(save_file)

curses.endwin()
