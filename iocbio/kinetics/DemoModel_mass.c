/* See http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator */
/*
A = A[**]
B = B[*]
C = C[**_*]
F1 : A + B <=> C
*/

/*

c_equations calculates the change in labeled species given an input of
            steady fluxes, constant pool sizes, and the current
            labeling state of all species.  Typically this function is
            used inside of a differential equation solver.
            
Input arguments:
       pool_list: Pool sizes for all metabolic species in the model.

       flux_list: Steady fluxes for all reactions in the model.  If
                  these are not steady your solver will complain ;)

     solver_time: The time provided by the differential equation
                  solver.  This can be used to change the default
                  labeling step change into a function of time.

      input_list: This is a list of the initial labeling state of all
                  mass isotopologue species.  The order is defined in
                  the code below.  An initial list is provided by the
                  user, and intermediate labeling states are provided
                  by the differential equation solver.
            
Output arguments:

             out: The time derivative of labeling state of all
                  species.  The order of this list is the same as the
                  input_list.
*/

void c_equations(double* pool_list, double* flux_list, double* solver_time, double* input_list, double* out)
{
double A_0 = input_list[0] ;
double A_1 = input_list[1] ;
double A_2 = input_list[2] ;
double B_0 = input_list[3] ;
double B_1 = input_list[4] ;
double C_0_0 = input_list[5] ;
double C_0_1 = input_list[6] ;
double C_1_0 = input_list[7] ;
double C_1_1 = input_list[8] ;
double C_2_0 = input_list[9] ;
double C_2_1 = input_list[10] ;

double fF1 = flux_list[0] ;
double rF1 = flux_list[1] ;

double pool_A = pool_list[0] ;
double pool_C = pool_list[1] ;
double pool_B = pool_list[2] ;

/*dA_0/dt=*/ out[0] = ( +rF1*(C_0_0+C_0_1)-fF1*((B_0+B_1)*A_0) )/ pool_A ;

/*dA_1/dt=*/ out[1] = ( +rF1*(C_1_0+C_1_1)-fF1*((B_0+B_1)*A_1) )/ pool_A ;

/*dA_2/dt=*/ out[2] = ( +rF1*(C_2_0+C_2_1)-fF1*((B_0+B_1)*A_2) )/ pool_A ;

/*dB_0/dt=*/ out[3] = ( +rF1*(C_0_0+C_1_0+C_2_0)-fF1*((A_0+A_1+A_2)*B_0) )/ pool_B ;

/*dB_1/dt=*/ out[4] = ( +rF1*(C_0_1+C_1_1+C_2_1)-fF1*((A_0+A_1+A_2)*B_1) )/ pool_B ;

/*dC_0_0/dt=*/ out[5] = ( +fF1*(A_0*B_0)-rF1*(C_0_0) )/ pool_C ;

/*dC_0_1/dt=*/ out[6] = ( +fF1*(A_0*B_1)-rF1*(C_0_1) )/ pool_C ;

/*dC_1_0/dt=*/ out[7] = ( +fF1*(A_1*B_0)-rF1*(C_1_0) )/ pool_C ;

/*dC_1_1/dt=*/ out[8] = ( +fF1*(A_1*B_1)-rF1*(C_1_1) )/ pool_C ;

/*dC_2_0/dt=*/ out[9] = ( +fF1*(A_2*B_0)-rF1*(C_2_0) )/ pool_C ;

/*dC_2_1/dt=*/ out[10] = ( +fF1*(A_2*B_1)-rF1*(C_2_1) )/ pool_C ;

} 
