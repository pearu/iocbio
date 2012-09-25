/* See http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator */
/*
A = A[*]
B = B[*]
C = C[**]
Ai = Ai[*]
Co = Co[**]
flux : A + B <=> C
in : A <= Ai
out : C => Co
*/

/*

c_equations calculates the change in labeled species given an input of steady fluxes, constant
            pool sizes, and the current labeling state of all species.  Typically this function 
            is used inside of a differential equation solver.
            
Inputs:
    pool_list:    Pool sizes for all metabolic species in the model.
    flux_list:    Steady fluxes for all reactions in the model.  If these are not steady your 
                  solver will complain ;)
    solver_time:  The time provided by the differential equation solver.  This can be used to 
                  change the default labeling step change into a function of time.
    input_list:   This is a list of the initial labeling state of all mass isotopologue species.
                  The order is defined in the code below.  An initial list is provided by the 
                  user, and intermediate labeling states are provided by the differential equation solver.
            
Output:
    out:          The updated labeling state of all species.  The order of this list is the same
                  as the input_list.
*/
void c_equations(double* pool_list, double* flux_list, double* solver_time, double* input_list, double* out)
{
double A_0 = input_list[0] ;
double A_1 = input_list[1] ;
double Ai_0 = input_list[2] ;
double Ai_1 = input_list[3] ;
double B_0 = input_list[4] ;
double B_1 = input_list[5] ;
double C_0 = input_list[6] ;
double C_1 = input_list[7] ;
double C_2 = input_list[8] ;
double Co_0 = input_list[9] ;
double Co_1 = input_list[10] ;
double Co_2 = input_list[11] ;

double fflux = flux_list[0] ;
double rflux = flux_list[1] ;
double fin = flux_list[2] ;
double fout = flux_list[3] ;

double pool_A = pool_list[0] ;
double pool_Ai = pool_list[1] ;
double pool_C = pool_list[2] ;
double pool_B = pool_list[3] ;
double pool_Co = pool_list[4] ;

/*dA_0/dt=*/ out[0] = ( +fin*(Ai_0)+rflux*(1/2.0*C_1+C_0)-fflux*((B_0+B_1)*A_0) )/ pool_A ;

/*dA_1/dt=*/ out[1] = ( +fin*(Ai_1)+rflux*(1/2.0*C_1+C_2)-fflux*((B_0+B_1)*A_1) )/ pool_A ;

/*dAi_0/dt=*/ out[2] = ( -fin*(Ai_0) )/ pool_Ai ;

/*dAi_1/dt=*/ out[3] = ( -fin*(Ai_1) )/ pool_Ai ;

/*dB_0/dt=*/ out[4] = ( +rflux*(1/2.0*C_1+C_0)-fflux*((A_0+A_1)*B_0) )/ pool_B ;

/*dB_1/dt=*/ out[5] = ( +rflux*(1/2.0*C_1+C_2)-fflux*((A_0+A_1)*B_1) )/ pool_B ;

/*dC_0/dt=*/ out[6] = ( +fflux*(A_0*B_0)-fout*(C_0)-rflux*(C_0) )/ pool_C ;

/*dC_1/dt=*/ out[7] = ( +fflux*(A_0*B_1+A_1*B_0)-fout*(C_1)-rflux*(C_1) )/ pool_C ;

/*dC_2/dt=*/ out[8] = ( +fflux*(A_1*B_1)-fout*(C_2)-rflux*(C_2) )/ pool_C ;

/*dCo_0/dt=*/ out[9] = ( +fout*(C_0) )/ pool_Co ;

/*dCo_1/dt=*/ out[10] = ( +fout*(C_1) )/ pool_Co ;

/*dCo_2/dt=*/ out[11] = ( +fout*(C_2) )/ pool_Co ;

} 
