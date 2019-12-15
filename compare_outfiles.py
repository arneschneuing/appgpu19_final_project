#!/usr/bin/python

import sys

TOL = 10e-5  # 10E-4-10E-6

input_args = sys.argv[1:]

if len(input_args) == 1 and input_args[0] == "-h":
    # Help
    print("Usage: python compare_outfiles.py file1 file2")
elif len(input_args) == 2:
    # Compare output files
    with open(input_args[0], 'r') as file1:
        with open(input_args[1], 'r') as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()
            assert len(lines1) == len(lines2)
            for i in range(len(lines1)):
                try:
                    value1 = float(lines1[i])
                    value2 = float(lines2[i])
                except:
                    continue
                
                if abs(value1 - value2) > TOL:
                    print("Outputs are different!")
                    exit()
            
            print("### Outputs are the same ###")
else:
    print("Invalid input argument(s).")
