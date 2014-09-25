import argparse
import numpy as np
import Functions.reading as reading
import phonopy.file_IO as file_IO
import Classes.controller as controller

#Define arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input file",type=str)
parser.add_argument("-wave_vector", help="Mother",type=int)

args = parser.parse_args()

#Reading from file
input_file = open(args.input_file , "r").readlines()
for i,line in enumerate(input_file):

    if "STRUCTURE FILE" in line:

        structure_file_name = input_file[i+1].replace('\n','')

    if "FORCE CONSTANTS" in line:
        force_constants_file_name = input_file[i+1].replace('\n','')


    if "PRIMITIVE MATRIX" in line:
        print('Primitive')
        primitive_matrix = [input_file[i+1].replace('\n','').split(),
                            input_file[i+2].replace('\n','').split(),
                            input_file[i+3].replace('\n','').split()]

        primitive_matrix = np.array(primitive_matrix,dtype=float)

    if "SUPERCELL MATRIX PHONOPY" in line:
        print('Super cell')
        super_cell_matrix = [input_file[i+1].replace('\n','').split(),
                             input_file[i+2].replace('\n','').split(),
                             input_file[i+3].replace('\n','').split()]

        super_cell_matrix = np.array(super_cell_matrix,dtype=int)


print(structure_file_name)
print(force_constants_file_name)
print(primitive_matrix)
print(super_cell_matrix)



