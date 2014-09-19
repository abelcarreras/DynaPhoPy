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



print(structure_file_name)
print(force_constants_file_name)




