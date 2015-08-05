# Introduction #

IOCBio provides a python module `builder.py` that constructs mass isotopologue equations.
Another module is provided `demo_model.py` that demonstrates how to use the builder
software. The system built by `oxygen_isotope_model.py` is studied in the following publication

  * David W. Schryer, Pearu Peterson, Ardo Illaste, Marko Vendelin. [Sensitivity analysis of flux determination in heart by H218O-provided  labeling using a dynamic isotopologue model of energy transfer pathways](http://dx.doi.org/10.1371/journal.pcbi.1002795). _PLoS Computational Biology_ 8(12):e1002795, 2012.

This page presents the software provided by this publication and
demonstrates its use.

# Usage #

First, download the source code and example files:

  * [builder.py](http://iocbio.googlecode.com/svn/trunk/iocbio/kinetics/builder.py) - Python code that provides the symbolic equation generator.

  * [demo\_model.py](http://iocbio.googlecode.com/svn/trunk/iocbio/kinetics/demo_model.py) - Minimal model to demonstrate the interface and illustrate how the equations are generated.

  * [oxygen\_isotope\_model.py](http://iocbio.googlecode.com/svn/trunk/iocbio/kinetics/oxygen_isotope_model.py) - Oxygen isotope network studied in the above paper.

Please examine the example files for instructions on how to build a custom mass isotopologue model.

To build the demonstration model, please make sure that
`builder.py` is in the same directory
as the example file `demo_model.py` and run the example file with a Python interpreter. The script outputs the equations generated:
```
$ python demo_model.py 

# See IsotopeModel.parse_system_string.__doc__ for syntax specifications.
#
# Definitions of species labeling
# <species name> = <species name>[<labeling pattern>]
#
A = A[**]
B = B[*]
C = C[**_*]

#
# Definitions of reactions
# <flux name> : <sum of reactants> (<=|=>|<=>) <sum of products>
#
F1 : A + B <=> C

Reactions split to elementary reactions:
  B+A <=rF1==fF1=> C
    B0+A00 -fF1-> C00_0
    B0+A01 -fF1-> C01_0
    B0+A10 -fF1-> C10_0
    B0+A11 -fF1-> C11_0
    B1+A00 -fF1-> C00_1
    B1+A01 -fF1-> C01_1
    B1+A10 -fF1-> C10_1
    B1+A11 -fF1-> C11_1
    C00_0 -rF1-> B0+A00
    C00_1 -rF1-> B1+A00
    C01_0 -rF1-> B0+A01
    C01_1 -rF1-> B1+A01
    C10_0 -rF1-> B0+A10
    C10_1 -rF1-> B1+A10
    C11_0 -rF1-> B0+A11
    C11_1 -rF1-> B1+A11
Kinetic equations from elementary reactions:
  A * dA00/dt = +rF1*(C00_0+C00_1)-fF1*(A00*B0+A00*B1)
  A * dA01/dt = +rF1*(C01_0+C01_1)-fF1*(A01*B0+A01*B1)
  A * dA10/dt = +rF1*(C10_0+C10_1)-fF1*(A10*B0+A10*B1)
  A * dA11/dt = +rF1*(C11_0+C11_1)-fF1*(A11*B0+A11*B1)
  B * dB0/dt = +rF1*(C00_0+C01_0+C10_0+C11_0)-fF1*(A00*B0+A01*B0+A10*B0+A11*B0)
  B * dB1/dt = +rF1*(C00_1+C01_1+C10_1+C11_1)-fF1*(A00*B1+A01*B1+A10*B1+A11*B1)
  C * dC00_0/dt = +fF1*(A00*B0)-rF1*(C00_0)
  C * dC00_1/dt = +fF1*(A00*B1)-rF1*(C00_1)
  C * dC01_0/dt = +fF1*(A01*B0)-rF1*(C01_0)
  C * dC01_1/dt = +fF1*(A01*B1)-rF1*(C01_1)
  C * dC10_0/dt = +fF1*(A10*B0)-rF1*(C10_0)
  C * dC10_1/dt = +fF1*(A10*B1)-rF1*(C10_1)
  C * dC11_0/dt = +fF1*(A11*B0)-rF1*(C11_0)
  C * dC11_1/dt = +fF1*(A11*B1)-rF1*(C11_1)
  where A, B, C are pool sizes.
Definitions of pool relations:
  A_0 = A00
  A_1 = A01+A10
  A_2 = A11
  B_0 = B0
  B_1 = B1
  C_0_0 = C00_0
  C_0_1 = C00_1
  C_1_0 = C01_0+C10_0
  C_1_1 = C01_1+C10_1
  C_2_0 = C11_0
  C_2_1 = C11_1
Time derivatives of pool relations with substituted elementary kinetic equations:
  A * dA_0/dt = +rF1*(C00_0+C00_1)-fF1*(A00*B0+A00*B1)
  A * dA_1/dt = +rF1*(C01_0+C01_1+C10_0+C10_1)-fF1*(A01*B0+A01*B1+A10*B0+A10*B1)
  A * dA_2/dt = +rF1*(C11_0+C11_1)-fF1*(A11*B0+A11*B1)
  B * dB_0/dt = +rF1*(C00_0+C01_0+C10_0+C11_0)-fF1*(A00*B0+A01*B0+A10*B0+A11*B0)
  B * dB_1/dt = +rF1*(C00_1+C01_1+C10_1+C11_1)-fF1*(A00*B1+A01*B1+A10*B1+A11*B1)
  C * dC_0_0/dt = +fF1*(A00*B0)-rF1*(C00_0)
  C * dC_0_1/dt = +fF1*(A00*B1)-rF1*(C00_1)
  C * dC_1_0/dt = +fF1*(A01*B0+A10*B0)-rF1*(C01_0+C10_0)
  C * dC_1_1/dt = +fF1*(A01*B1+A10*B1)-rF1*(C01_1+C10_1)
  C * dC_2_0/dt = +fF1*(A11*B0)-rF1*(C11_0)
  C * dC_2_1/dt = +fF1*(A11*B1)-rF1*(C11_1)
Mass isotopomer kinetic equations with substituded pool relations:
  A * dA_0/dt = +rF1*(C_0_0+C_0_1)-fF1*((B_0+B_1)*A_0)
  A * dA_1/dt = +rF1*(C_1_0+C_1_1)-fF1*((B_0+B_1)*A_1)
  A * dA_2/dt = +rF1*(C_2_0+C_2_1)-fF1*((B_0+B_1)*A_2)
  B * dB_0/dt = +rF1*(C_0_0+C_1_0+C_2_0)-fF1*((A_0+A_1+A_2)*B_0)
  B * dB_1/dt = +rF1*(C_0_1+C_1_1+C_2_1)-fF1*((A_0+A_1+A_2)*B_1)
  C * dC_0_0/dt = +fF1*(A_0*B_0)-rF1*(C_0_0)
  C * dC_0_1/dt = +fF1*(A_0*B_1)-rF1*(C_0_1)
  C * dC_1_0/dt = +fF1*(A_1*B_0)-rF1*(C_1_0)
  C * dC_1_1/dt = +fF1*(A_1*B_1)-rF1*(C_1_1)
  C * dC_2_0/dt = +fF1*(A_2*B_0)-rF1*(C_2_0)
  C * dC_2_1/dt = +fF1*(A_2*B_1)-rF1*(C_2_1)
```

To create the model studied in the above paper, please make sure that `builder.py` script is in the same directory as the example file `oxygen_isotope_model.py` and run the example file with a Python interpreter. One can examine the contents of the files generated to view the equations.  The output from this script only describes what the algorithm is doing:
```
$ python oxygen_isotope_model.py
Generating isotopologue equations:
Performing first term collection.
Performing second term collection.
Generating mass isotopologue equations:
Applying pool relations.
Performing first term collection.
Performing second term collection.
Performing third term collection.
Performing final term collection.

Wrote:  model_3000_mass.c
```
The expected output file can be found [here](http://iocbio.googlecode.com/svn/trunk/iocbio/kinetics/model_3000_mass.c) that contains a C function of mass isotopomer kinetic equations, see the header of the file for documentation.