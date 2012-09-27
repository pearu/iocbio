/* See http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator */
/*
ADPs+ADPm+ADPi+ADPo+ADPe = ADPs[***]+ADPm[***]+ADPi[***]+ADPo[***]+ADPe[***]
Ps+Pm+Pe+Po = Ps[****]+Pm[****]+Pe[****]+Po[****]
Wo+Ws+We = Wo[*]+Ws[*]+We[*]
CPi+CPo = CPi[***]+CPo[***]
ATPs+ATPm+ATPo+ATPe+ATPi = ATPs[***_***]+ATPm[***_***]+ATPo[***_***]+ATPe[***_***]+ATPi[***_***]
ADPms : ADPm <=> ADPs
Pms   : Pm <=> Ps
Wos   : Wo <=> Ws
ASs   : ADPs + Ps <=> ATPs + Ws
ATPsm : ATPs => ATPm
ATPoe : ATPo => ATPe
Peo   : Pe <=> Po
Weo   : We <=> Wo
ASe   : ATPe + We <=> ADPe + Pe
ADPeo : ADPe <=> ADPo
AKi  : 2 ADPi <=> ATPi
AKo  : ATPo <=> 2 ADPo
CKi   : ATPi <=> ADPi + CPi
CKo   : CPo + ADPo <=> ATPo 
ADPim : ADPi <=> ADPm
ADPoi : ADPo <=> ADPi 
ATPmi : ATPm <=> ATPi
ATPio : ATPi <=> ATPo
Cio   : CPi <=> CPo
Pom   : Po => Pm
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
double ADPe_0 = input_list[0] ;
double ADPe_1 = input_list[1] ;
double ADPe_2 = input_list[2] ;
double ADPe_3 = input_list[3] ;
double ADPi_0 = input_list[4] ;
double ADPi_1 = input_list[5] ;
double ADPi_2 = input_list[6] ;
double ADPi_3 = input_list[7] ;
double ADPm_0 = input_list[8] ;
double ADPm_1 = input_list[9] ;
double ADPm_2 = input_list[10] ;
double ADPm_3 = input_list[11] ;
double ADPo_0 = input_list[12] ;
double ADPo_1 = input_list[13] ;
double ADPo_2 = input_list[14] ;
double ADPo_3 = input_list[15] ;
double ADPs_0 = input_list[16] ;
double ADPs_1 = input_list[17] ;
double ADPs_2 = input_list[18] ;
double ADPs_3 = input_list[19] ;
double ATPe_0_0 = input_list[20] ;
double ATPe_0_1 = input_list[21] ;
double ATPe_0_2 = input_list[22] ;
double ATPe_0_3 = input_list[23] ;
double ATPe_1_0 = input_list[24] ;
double ATPe_1_1 = input_list[25] ;
double ATPe_1_2 = input_list[26] ;
double ATPe_1_3 = input_list[27] ;
double ATPe_2_0 = input_list[28] ;
double ATPe_2_1 = input_list[29] ;
double ATPe_2_2 = input_list[30] ;
double ATPe_2_3 = input_list[31] ;
double ATPe_3_0 = input_list[32] ;
double ATPe_3_1 = input_list[33] ;
double ATPe_3_2 = input_list[34] ;
double ATPe_3_3 = input_list[35] ;
double ATPi_0_0 = input_list[36] ;
double ATPi_0_1 = input_list[37] ;
double ATPi_0_2 = input_list[38] ;
double ATPi_0_3 = input_list[39] ;
double ATPi_1_0 = input_list[40] ;
double ATPi_1_1 = input_list[41] ;
double ATPi_1_2 = input_list[42] ;
double ATPi_1_3 = input_list[43] ;
double ATPi_2_0 = input_list[44] ;
double ATPi_2_1 = input_list[45] ;
double ATPi_2_2 = input_list[46] ;
double ATPi_2_3 = input_list[47] ;
double ATPi_3_0 = input_list[48] ;
double ATPi_3_1 = input_list[49] ;
double ATPi_3_2 = input_list[50] ;
double ATPi_3_3 = input_list[51] ;
double ATPm_0_0 = input_list[52] ;
double ATPm_0_1 = input_list[53] ;
double ATPm_0_2 = input_list[54] ;
double ATPm_0_3 = input_list[55] ;
double ATPm_1_0 = input_list[56] ;
double ATPm_1_1 = input_list[57] ;
double ATPm_1_2 = input_list[58] ;
double ATPm_1_3 = input_list[59] ;
double ATPm_2_0 = input_list[60] ;
double ATPm_2_1 = input_list[61] ;
double ATPm_2_2 = input_list[62] ;
double ATPm_2_3 = input_list[63] ;
double ATPm_3_0 = input_list[64] ;
double ATPm_3_1 = input_list[65] ;
double ATPm_3_2 = input_list[66] ;
double ATPm_3_3 = input_list[67] ;
double ATPo_0_0 = input_list[68] ;
double ATPo_0_1 = input_list[69] ;
double ATPo_0_2 = input_list[70] ;
double ATPo_0_3 = input_list[71] ;
double ATPo_1_0 = input_list[72] ;
double ATPo_1_1 = input_list[73] ;
double ATPo_1_2 = input_list[74] ;
double ATPo_1_3 = input_list[75] ;
double ATPo_2_0 = input_list[76] ;
double ATPo_2_1 = input_list[77] ;
double ATPo_2_2 = input_list[78] ;
double ATPo_2_3 = input_list[79] ;
double ATPo_3_0 = input_list[80] ;
double ATPo_3_1 = input_list[81] ;
double ATPo_3_2 = input_list[82] ;
double ATPo_3_3 = input_list[83] ;
double ATPs_0_0 = input_list[84] ;
double ATPs_0_1 = input_list[85] ;
double ATPs_0_2 = input_list[86] ;
double ATPs_0_3 = input_list[87] ;
double ATPs_1_0 = input_list[88] ;
double ATPs_1_1 = input_list[89] ;
double ATPs_1_2 = input_list[90] ;
double ATPs_1_3 = input_list[91] ;
double ATPs_2_0 = input_list[92] ;
double ATPs_2_1 = input_list[93] ;
double ATPs_2_2 = input_list[94] ;
double ATPs_2_3 = input_list[95] ;
double ATPs_3_0 = input_list[96] ;
double ATPs_3_1 = input_list[97] ;
double ATPs_3_2 = input_list[98] ;
double ATPs_3_3 = input_list[99] ;
double CPi_0 = input_list[100] ;
double CPi_1 = input_list[101] ;
double CPi_2 = input_list[102] ;
double CPi_3 = input_list[103] ;
double CPo_0 = input_list[104] ;
double CPo_1 = input_list[105] ;
double CPo_2 = input_list[106] ;
double CPo_3 = input_list[107] ;
double Pe_0 = input_list[108] ;
double Pe_1 = input_list[109] ;
double Pe_2 = input_list[110] ;
double Pe_3 = input_list[111] ;
double Pe_4 = input_list[112] ;
double Pm_0 = input_list[113] ;
double Pm_1 = input_list[114] ;
double Pm_2 = input_list[115] ;
double Pm_3 = input_list[116] ;
double Pm_4 = input_list[117] ;
double Po_0 = input_list[118] ;
double Po_1 = input_list[119] ;
double Po_2 = input_list[120] ;
double Po_3 = input_list[121] ;
double Po_4 = input_list[122] ;
double Ps_0 = input_list[123] ;
double Ps_1 = input_list[124] ;
double Ps_2 = input_list[125] ;
double Ps_3 = input_list[126] ;
double Ps_4 = input_list[127] ;
double We_0 = input_list[128] ;
double We_1 = input_list[129] ;
double Wo_0 = 0.7 ;
double Wo_1 = 0.3 ;
double Ws_0 = input_list[130] ;
double Ws_1 = input_list[131] ;

double fADPeo = flux_list[0] ;
double rADPeo = flux_list[1] ;
double fADPim = flux_list[2] ;
double rADPim = flux_list[3] ;
double fADPms = flux_list[4] ;
double rADPms = flux_list[5] ;
double fADPoi = flux_list[6] ;
double rADPoi = flux_list[7] ;
double fAKi = flux_list[8] ;
double rAKi = flux_list[9] ;
double fAKo = flux_list[10] ;
double rAKo = flux_list[11] ;
double fASe = flux_list[12] ;
double rASe = flux_list[13] ;
double fASs = flux_list[14] ;
double rASs = flux_list[15] ;
double fATPio = flux_list[16] ;
double rATPio = flux_list[17] ;
double fATPmi = flux_list[18] ;
double rATPmi = flux_list[19] ;
double fATPoe = flux_list[20] ;
double fATPsm = flux_list[21] ;
double fCKi = flux_list[22] ;
double rCKi = flux_list[23] ;
double fCKo = flux_list[24] ;
double rCKo = flux_list[25] ;
double fCio = flux_list[26] ;
double rCio = flux_list[27] ;
double fPeo = flux_list[28] ;
double rPeo = flux_list[29] ;
double fPms = flux_list[30] ;
double rPms = flux_list[31] ;
double fPom = flux_list[32] ;
double fWeo = flux_list[33] ;
double rWeo = flux_list[34] ;
double fWos = flux_list[35] ;
double rWos = flux_list[36] ;

double pool_CPi = pool_list[0] ;
double pool_ADPi = pool_list[1] ;
double pool_ADPo = pool_list[2] ;
double pool_ADPm = pool_list[3] ;
double pool_CPo = pool_list[4] ;
double pool_ATPs = pool_list[5] ;
double pool_We = pool_list[6] ;
double pool_ADPe = pool_list[7] ;
double pool_Ps = pool_list[8] ;
double pool_ATPi = pool_list[9] ;
double pool_ATPo = pool_list[10] ;
double pool_ATPm = pool_list[11] ;
double pool_Pe = pool_list[12] ;
double pool_ADPs = pool_list[13] ;
double pool_Ws = pool_list[14] ;
double pool_Po = pool_list[15] ;
double pool_ATPe = pool_list[16] ;
double pool_Pm = pool_list[17] ;

/*dADPe_0/dt=*/ out[0] = ( +fASe*((ATPe_0_0+ATPe_0_1+ATPe_0_2+ATPe_0_3)*(We_0+We_1))+rADPeo*(ADPo_0)-fADPeo*(ADPe_0)-rASe*((Pe_0+Pe_1+Pe_2+Pe_3+Pe_4)*ADPe_0) )/ pool_ADPe ;

/*dADPe_1/dt=*/ out[1] = ( +fASe*((ATPe_1_0+ATPe_1_1+ATPe_1_2+ATPe_1_3)*(We_0+We_1))+rADPeo*(ADPo_1)-fADPeo*(ADPe_1)-rASe*((Pe_0+Pe_1+Pe_2+Pe_3+Pe_4)*ADPe_1) )/ pool_ADPe ;

/*dADPe_2/dt=*/ out[2] = ( +fASe*((ATPe_2_0+ATPe_2_1+ATPe_2_2+ATPe_2_3)*(We_0+We_1))+rADPeo*(ADPo_2)-fADPeo*(ADPe_2)-rASe*((Pe_0+Pe_1+Pe_2+Pe_3+Pe_4)*ADPe_2) )/ pool_ADPe ;

/*dADPe_3/dt=*/ out[3] = ( +fASe*((ATPe_3_0+ATPe_3_1+ATPe_3_2+ATPe_3_3)*(We_0+We_1))+rADPeo*(ADPo_3)-fADPeo*(ADPe_3)-rASe*((Pe_0+Pe_1+Pe_2+Pe_3+Pe_4)*ADPe_3) )/ pool_ADPe ;

/*dADPi_0/dt=*/ out[4] = ( +fADPoi*(ADPo_0)+fCKi*(ATPi_0_0+ATPi_0_1+ATPi_0_2+ATPi_0_3)+rADPim*(ADPm_0)+rAKi*(2*ATPi_0_0+ATPi_0_1+ATPi_0_2+ATPi_0_3+ATPi_1_0+ATPi_2_0+ATPi_3_0)-fADPim*(ADPi_0)-fAKi*(2*(ADPi_0+ADPi_1+ADPi_2+ADPi_3)*ADPi_0)-rADPoi*(ADPi_0)-rCKi*((CPi_0+CPi_1+CPi_2+CPi_3)*ADPi_0) )/ pool_ADPi ;

/*dADPi_1/dt=*/ out[5] = ( +fADPoi*(ADPo_1)+fCKi*(ATPi_1_0+ATPi_1_1+ATPi_1_2+ATPi_1_3)+rADPim*(ADPm_1)+rAKi*(2*ATPi_1_1+ATPi_0_1+ATPi_1_0+ATPi_1_2+ATPi_1_3+ATPi_2_1+ATPi_3_1)-fADPim*(ADPi_1)-fAKi*(2*(ADPi_0+ADPi_1+ADPi_2+ADPi_3)*ADPi_1)-rADPoi*(ADPi_1)-rCKi*((CPi_0+CPi_1+CPi_2+CPi_3)*ADPi_1) )/ pool_ADPi ;

/*dADPi_2/dt=*/ out[6] = ( +fADPoi*(ADPo_2)+fCKi*(ATPi_2_0+ATPi_2_1+ATPi_2_2+ATPi_2_3)+rADPim*(ADPm_2)+rAKi*(2*ATPi_2_2+ATPi_0_2+ATPi_1_2+ATPi_2_0+ATPi_2_1+ATPi_2_3+ATPi_3_2)-fADPim*(ADPi_2)-fAKi*(2*(ADPi_0+ADPi_1+ADPi_2+ADPi_3)*ADPi_2)-rADPoi*(ADPi_2)-rCKi*((CPi_0+CPi_1+CPi_2+CPi_3)*ADPi_2) )/ pool_ADPi ;

/*dADPi_3/dt=*/ out[7] = ( +fADPoi*(ADPo_3)+fCKi*(ATPi_3_0+ATPi_3_1+ATPi_3_2+ATPi_3_3)+rADPim*(ADPm_3)+rAKi*(2*ATPi_3_3+ATPi_0_3+ATPi_1_3+ATPi_2_3+ATPi_3_0+ATPi_3_1+ATPi_3_2)-fADPim*(ADPi_3)-fAKi*(2*(ADPi_0+ADPi_1+ADPi_2+ADPi_3)*ADPi_3)-rADPoi*(ADPi_3)-rCKi*((CPi_0+CPi_1+CPi_2+CPi_3)*ADPi_3) )/ pool_ADPi ;

/*dADPm_0/dt=*/ out[8] = ( +fADPim*(ADPi_0)+rADPms*(ADPs_0)-fADPms*(ADPm_0)-rADPim*(ADPm_0) )/ pool_ADPm ;

/*dADPm_1/dt=*/ out[9] = ( +fADPim*(ADPi_1)+rADPms*(ADPs_1)-fADPms*(ADPm_1)-rADPim*(ADPm_1) )/ pool_ADPm ;

/*dADPm_2/dt=*/ out[10] = ( +fADPim*(ADPi_2)+rADPms*(ADPs_2)-fADPms*(ADPm_2)-rADPim*(ADPm_2) )/ pool_ADPm ;

/*dADPm_3/dt=*/ out[11] = ( +fADPim*(ADPi_3)+rADPms*(ADPs_3)-fADPms*(ADPm_3)-rADPim*(ADPm_3) )/ pool_ADPm ;

/*dADPo_0/dt=*/ out[12] = ( +fADPeo*(ADPe_0)+fAKo*(2*ATPo_0_0+ATPo_0_1+ATPo_0_2+ATPo_0_3+ATPo_1_0+ATPo_2_0+ATPo_3_0)+rADPoi*(ADPi_0)+rCKo*(ATPo_0_0+ATPo_0_1+ATPo_0_2+ATPo_0_3)-fADPoi*(ADPo_0)-fCKo*((CPo_0+CPo_1+CPo_2+CPo_3)*ADPo_0)-rADPeo*(ADPo_0)-rAKo*(2*(ADPo_0+ADPo_1+ADPo_2+ADPo_3)*ADPo_0) )/ pool_ADPo ;

/*dADPo_1/dt=*/ out[13] = ( +fADPeo*(ADPe_1)+fAKo*(2*ATPo_1_1+ATPo_0_1+ATPo_1_0+ATPo_1_2+ATPo_1_3+ATPo_2_1+ATPo_3_1)+rADPoi*(ADPi_1)+rCKo*(ATPo_1_0+ATPo_1_1+ATPo_1_2+ATPo_1_3)-fADPoi*(ADPo_1)-fCKo*((CPo_0+CPo_1+CPo_2+CPo_3)*ADPo_1)-rADPeo*(ADPo_1)-rAKo*(2*(ADPo_0+ADPo_1+ADPo_2+ADPo_3)*ADPo_1) )/ pool_ADPo ;

/*dADPo_2/dt=*/ out[14] = ( +fADPeo*(ADPe_2)+fAKo*(2*ATPo_2_2+ATPo_0_2+ATPo_1_2+ATPo_2_0+ATPo_2_1+ATPo_2_3+ATPo_3_2)+rADPoi*(ADPi_2)+rCKo*(ATPo_2_0+ATPo_2_1+ATPo_2_2+ATPo_2_3)-fADPoi*(ADPo_2)-fCKo*((CPo_0+CPo_1+CPo_2+CPo_3)*ADPo_2)-rADPeo*(ADPo_2)-rAKo*(2*(ADPo_0+ADPo_1+ADPo_2+ADPo_3)*ADPo_2) )/ pool_ADPo ;

/*dADPo_3/dt=*/ out[15] = ( +fADPeo*(ADPe_3)+fAKo*(2*ATPo_3_3+ATPo_0_3+ATPo_1_3+ATPo_2_3+ATPo_3_0+ATPo_3_1+ATPo_3_2)+rADPoi*(ADPi_3)+rCKo*(ATPo_3_0+ATPo_3_1+ATPo_3_2+ATPo_3_3)-fADPoi*(ADPo_3)-fCKo*((CPo_0+CPo_1+CPo_2+CPo_3)*ADPo_3)-rADPeo*(ADPo_3)-rAKo*(2*(ADPo_0+ADPo_1+ADPo_2+ADPo_3)*ADPo_3) )/ pool_ADPo ;

/*dADPs_0/dt=*/ out[16] = ( +fADPms*(ADPm_0)+rASs*((ATPs_0_0+ATPs_0_1+ATPs_0_2+ATPs_0_3)*(Ws_0+Ws_1))-fASs*((Ps_0+Ps_1+Ps_2+Ps_3+Ps_4)*ADPs_0)-rADPms*(ADPs_0) )/ pool_ADPs ;

/*dADPs_1/dt=*/ out[17] = ( +fADPms*(ADPm_1)+rASs*((ATPs_1_0+ATPs_1_1+ATPs_1_2+ATPs_1_3)*(Ws_0+Ws_1))-fASs*((Ps_0+Ps_1+Ps_2+Ps_3+Ps_4)*ADPs_1)-rADPms*(ADPs_1) )/ pool_ADPs ;

/*dADPs_2/dt=*/ out[18] = ( +fADPms*(ADPm_2)+rASs*((ATPs_2_0+ATPs_2_1+ATPs_2_2+ATPs_2_3)*(Ws_0+Ws_1))-fASs*((Ps_0+Ps_1+Ps_2+Ps_3+Ps_4)*ADPs_2)-rADPms*(ADPs_2) )/ pool_ADPs ;

/*dADPs_3/dt=*/ out[19] = ( +fADPms*(ADPm_3)+rASs*((ATPs_3_0+ATPs_3_1+ATPs_3_2+ATPs_3_3)*(Ws_0+Ws_1))-fASs*((Ps_0+Ps_1+Ps_2+Ps_3+Ps_4)*ADPs_3)-rADPms*(ADPs_3) )/ pool_ADPs ;

/*dATPe_0_0/dt=*/ out[20] = ( +fATPoe*(ATPo_0_0)+rASe*((1/4.0*Pe_1+Pe_0)*ADPe_0)-fASe*((We_0+We_1)*ATPe_0_0) )/ pool_ATPe ;

/*dATPe_0_1/dt=*/ out[21] = ( +fATPoe*(ATPo_0_1)+rASe*((1/2.0*Pe_2+3/4.0*Pe_1)*ADPe_0)-fASe*((We_0+We_1)*ATPe_0_1) )/ pool_ATPe ;

/*dATPe_0_2/dt=*/ out[22] = ( +fATPoe*(ATPo_0_2)+rASe*((1/2.0*Pe_2+3/4.0*Pe_3)*ADPe_0)-fASe*((We_0+We_1)*ATPe_0_2) )/ pool_ATPe ;

/*dATPe_0_3/dt=*/ out[23] = ( +fATPoe*(ATPo_0_3)+rASe*((1/4.0*Pe_3+Pe_4)*ADPe_0)-fASe*((We_0+We_1)*ATPe_0_3) )/ pool_ATPe ;

/*dATPe_1_0/dt=*/ out[24] = ( +fATPoe*(ATPo_1_0)+rASe*((1/4.0*Pe_1+Pe_0)*ADPe_1)-fASe*((We_0+We_1)*ATPe_1_0) )/ pool_ATPe ;

/*dATPe_1_1/dt=*/ out[25] = ( +fATPoe*(ATPo_1_1)+rASe*((1/2.0*Pe_2+3/4.0*Pe_1)*ADPe_1)-fASe*((We_0+We_1)*ATPe_1_1) )/ pool_ATPe ;

/*dATPe_1_2/dt=*/ out[26] = ( +fATPoe*(ATPo_1_2)+rASe*((1/2.0*Pe_2+3/4.0*Pe_3)*ADPe_1)-fASe*((We_0+We_1)*ATPe_1_2) )/ pool_ATPe ;

/*dATPe_1_3/dt=*/ out[27] = ( +fATPoe*(ATPo_1_3)+rASe*((1/4.0*Pe_3+Pe_4)*ADPe_1)-fASe*((We_0+We_1)*ATPe_1_3) )/ pool_ATPe ;

/*dATPe_2_0/dt=*/ out[28] = ( +fATPoe*(ATPo_2_0)+rASe*((1/4.0*Pe_1+Pe_0)*ADPe_2)-fASe*((We_0+We_1)*ATPe_2_0) )/ pool_ATPe ;

/*dATPe_2_1/dt=*/ out[29] = ( +fATPoe*(ATPo_2_1)+rASe*((1/2.0*Pe_2+3/4.0*Pe_1)*ADPe_2)-fASe*((We_0+We_1)*ATPe_2_1) )/ pool_ATPe ;

/*dATPe_2_2/dt=*/ out[30] = ( +fATPoe*(ATPo_2_2)+rASe*((1/2.0*Pe_2+3/4.0*Pe_3)*ADPe_2)-fASe*((We_0+We_1)*ATPe_2_2) )/ pool_ATPe ;

/*dATPe_2_3/dt=*/ out[31] = ( +fATPoe*(ATPo_2_3)+rASe*((1/4.0*Pe_3+Pe_4)*ADPe_2)-fASe*((We_0+We_1)*ATPe_2_3) )/ pool_ATPe ;

/*dATPe_3_0/dt=*/ out[32] = ( +fATPoe*(ATPo_3_0)+rASe*((1/4.0*Pe_1+Pe_0)*ADPe_3)-fASe*((We_0+We_1)*ATPe_3_0) )/ pool_ATPe ;

/*dATPe_3_1/dt=*/ out[33] = ( +fATPoe*(ATPo_3_1)+rASe*((1/2.0*Pe_2+3/4.0*Pe_1)*ADPe_3)-fASe*((We_0+We_1)*ATPe_3_1) )/ pool_ATPe ;

/*dATPe_3_2/dt=*/ out[34] = ( +fATPoe*(ATPo_3_2)+rASe*((1/2.0*Pe_2+3/4.0*Pe_3)*ADPe_3)-fASe*((We_0+We_1)*ATPe_3_2) )/ pool_ATPe ;

/*dATPe_3_3/dt=*/ out[35] = ( +fATPoe*(ATPo_3_3)+rASe*((1/4.0*Pe_3+Pe_4)*ADPe_3)-fASe*((We_0+We_1)*ATPe_3_3) )/ pool_ATPe ;

/*dATPi_0_0/dt=*/ out[36] = ( +fAKi*(ADPi_0*ADPi_0)+fATPmi*(ATPm_0_0)+rATPio*(ATPo_0_0)+rCKi*(ADPi_0*CPi_0)-fATPio*(ATPi_0_0)-fCKi*(ATPi_0_0)-rAKi*(ATPi_0_0)-rATPmi*(ATPi_0_0) )/ pool_ATPi ;

/*dATPi_0_1/dt=*/ out[37] = ( +fAKi*(ADPi_0*ADPi_1)+fATPmi*(ATPm_0_1)+rATPio*(ATPo_0_1)+rCKi*(ADPi_0*CPi_1)-fATPio*(ATPi_0_1)-fCKi*(ATPi_0_1)-rAKi*(ATPi_0_1)-rATPmi*(ATPi_0_1) )/ pool_ATPi ;

/*dATPi_0_2/dt=*/ out[38] = ( +fAKi*(ADPi_0*ADPi_2)+fATPmi*(ATPm_0_2)+rATPio*(ATPo_0_2)+rCKi*(ADPi_0*CPi_2)-fATPio*(ATPi_0_2)-fCKi*(ATPi_0_2)-rAKi*(ATPi_0_2)-rATPmi*(ATPi_0_2) )/ pool_ATPi ;

/*dATPi_0_3/dt=*/ out[39] = ( +fAKi*(ADPi_0*ADPi_3)+fATPmi*(ATPm_0_3)+rATPio*(ATPo_0_3)+rCKi*(ADPi_0*CPi_3)-fATPio*(ATPi_0_3)-fCKi*(ATPi_0_3)-rAKi*(ATPi_0_3)-rATPmi*(ATPi_0_3) )/ pool_ATPi ;

/*dATPi_1_0/dt=*/ out[40] = ( +fAKi*(ADPi_0*ADPi_1)+fATPmi*(ATPm_1_0)+rATPio*(ATPo_1_0)+rCKi*(ADPi_1*CPi_0)-fATPio*(ATPi_1_0)-fCKi*(ATPi_1_0)-rAKi*(ATPi_1_0)-rATPmi*(ATPi_1_0) )/ pool_ATPi ;

/*dATPi_1_1/dt=*/ out[41] = ( +fAKi*(ADPi_1*ADPi_1)+fATPmi*(ATPm_1_1)+rATPio*(ATPo_1_1)+rCKi*(ADPi_1*CPi_1)-fATPio*(ATPi_1_1)-fCKi*(ATPi_1_1)-rAKi*(ATPi_1_1)-rATPmi*(ATPi_1_1) )/ pool_ATPi ;

/*dATPi_1_2/dt=*/ out[42] = ( +fAKi*(ADPi_1*ADPi_2)+fATPmi*(ATPm_1_2)+rATPio*(ATPo_1_2)+rCKi*(ADPi_1*CPi_2)-fATPio*(ATPi_1_2)-fCKi*(ATPi_1_2)-rAKi*(ATPi_1_2)-rATPmi*(ATPi_1_2) )/ pool_ATPi ;

/*dATPi_1_3/dt=*/ out[43] = ( +fAKi*(ADPi_1*ADPi_3)+fATPmi*(ATPm_1_3)+rATPio*(ATPo_1_3)+rCKi*(ADPi_1*CPi_3)-fATPio*(ATPi_1_3)-fCKi*(ATPi_1_3)-rAKi*(ATPi_1_3)-rATPmi*(ATPi_1_3) )/ pool_ATPi ;

/*dATPi_2_0/dt=*/ out[44] = ( +fAKi*(ADPi_0*ADPi_2)+fATPmi*(ATPm_2_0)+rATPio*(ATPo_2_0)+rCKi*(ADPi_2*CPi_0)-fATPio*(ATPi_2_0)-fCKi*(ATPi_2_0)-rAKi*(ATPi_2_0)-rATPmi*(ATPi_2_0) )/ pool_ATPi ;

/*dATPi_2_1/dt=*/ out[45] = ( +fAKi*(ADPi_1*ADPi_2)+fATPmi*(ATPm_2_1)+rATPio*(ATPo_2_1)+rCKi*(ADPi_2*CPi_1)-fATPio*(ATPi_2_1)-fCKi*(ATPi_2_1)-rAKi*(ATPi_2_1)-rATPmi*(ATPi_2_1) )/ pool_ATPi ;

/*dATPi_2_2/dt=*/ out[46] = ( +fAKi*(ADPi_2*ADPi_2)+fATPmi*(ATPm_2_2)+rATPio*(ATPo_2_2)+rCKi*(ADPi_2*CPi_2)-fATPio*(ATPi_2_2)-fCKi*(ATPi_2_2)-rAKi*(ATPi_2_2)-rATPmi*(ATPi_2_2) )/ pool_ATPi ;

/*dATPi_2_3/dt=*/ out[47] = ( +fAKi*(ADPi_2*ADPi_3)+fATPmi*(ATPm_2_3)+rATPio*(ATPo_2_3)+rCKi*(ADPi_2*CPi_3)-fATPio*(ATPi_2_3)-fCKi*(ATPi_2_3)-rAKi*(ATPi_2_3)-rATPmi*(ATPi_2_3) )/ pool_ATPi ;

/*dATPi_3_0/dt=*/ out[48] = ( +fAKi*(ADPi_0*ADPi_3)+fATPmi*(ATPm_3_0)+rATPio*(ATPo_3_0)+rCKi*(ADPi_3*CPi_0)-fATPio*(ATPi_3_0)-fCKi*(ATPi_3_0)-rAKi*(ATPi_3_0)-rATPmi*(ATPi_3_0) )/ pool_ATPi ;

/*dATPi_3_1/dt=*/ out[49] = ( +fAKi*(ADPi_1*ADPi_3)+fATPmi*(ATPm_3_1)+rATPio*(ATPo_3_1)+rCKi*(ADPi_3*CPi_1)-fATPio*(ATPi_3_1)-fCKi*(ATPi_3_1)-rAKi*(ATPi_3_1)-rATPmi*(ATPi_3_1) )/ pool_ATPi ;

/*dATPi_3_2/dt=*/ out[50] = ( +fAKi*(ADPi_2*ADPi_3)+fATPmi*(ATPm_3_2)+rATPio*(ATPo_3_2)+rCKi*(ADPi_3*CPi_2)-fATPio*(ATPi_3_2)-fCKi*(ATPi_3_2)-rAKi*(ATPi_3_2)-rATPmi*(ATPi_3_2) )/ pool_ATPi ;

/*dATPi_3_3/dt=*/ out[51] = ( +fAKi*(ADPi_3*ADPi_3)+fATPmi*(ATPm_3_3)+rATPio*(ATPo_3_3)+rCKi*(ADPi_3*CPi_3)-fATPio*(ATPi_3_3)-fCKi*(ATPi_3_3)-rAKi*(ATPi_3_3)-rATPmi*(ATPi_3_3) )/ pool_ATPi ;

/*dATPm_0_0/dt=*/ out[52] = ( +fATPsm*(ATPs_0_0)+rATPmi*(ATPi_0_0)-fATPmi*(ATPm_0_0) )/ pool_ATPm ;

/*dATPm_0_1/dt=*/ out[53] = ( +fATPsm*(ATPs_0_1)+rATPmi*(ATPi_0_1)-fATPmi*(ATPm_0_1) )/ pool_ATPm ;

/*dATPm_0_2/dt=*/ out[54] = ( +fATPsm*(ATPs_0_2)+rATPmi*(ATPi_0_2)-fATPmi*(ATPm_0_2) )/ pool_ATPm ;

/*dATPm_0_3/dt=*/ out[55] = ( +fATPsm*(ATPs_0_3)+rATPmi*(ATPi_0_3)-fATPmi*(ATPm_0_3) )/ pool_ATPm ;

/*dATPm_1_0/dt=*/ out[56] = ( +fATPsm*(ATPs_1_0)+rATPmi*(ATPi_1_0)-fATPmi*(ATPm_1_0) )/ pool_ATPm ;

/*dATPm_1_1/dt=*/ out[57] = ( +fATPsm*(ATPs_1_1)+rATPmi*(ATPi_1_1)-fATPmi*(ATPm_1_1) )/ pool_ATPm ;

/*dATPm_1_2/dt=*/ out[58] = ( +fATPsm*(ATPs_1_2)+rATPmi*(ATPi_1_2)-fATPmi*(ATPm_1_2) )/ pool_ATPm ;

/*dATPm_1_3/dt=*/ out[59] = ( +fATPsm*(ATPs_1_3)+rATPmi*(ATPi_1_3)-fATPmi*(ATPm_1_3) )/ pool_ATPm ;

/*dATPm_2_0/dt=*/ out[60] = ( +fATPsm*(ATPs_2_0)+rATPmi*(ATPi_2_0)-fATPmi*(ATPm_2_0) )/ pool_ATPm ;

/*dATPm_2_1/dt=*/ out[61] = ( +fATPsm*(ATPs_2_1)+rATPmi*(ATPi_2_1)-fATPmi*(ATPm_2_1) )/ pool_ATPm ;

/*dATPm_2_2/dt=*/ out[62] = ( +fATPsm*(ATPs_2_2)+rATPmi*(ATPi_2_2)-fATPmi*(ATPm_2_2) )/ pool_ATPm ;

/*dATPm_2_3/dt=*/ out[63] = ( +fATPsm*(ATPs_2_3)+rATPmi*(ATPi_2_3)-fATPmi*(ATPm_2_3) )/ pool_ATPm ;

/*dATPm_3_0/dt=*/ out[64] = ( +fATPsm*(ATPs_3_0)+rATPmi*(ATPi_3_0)-fATPmi*(ATPm_3_0) )/ pool_ATPm ;

/*dATPm_3_1/dt=*/ out[65] = ( +fATPsm*(ATPs_3_1)+rATPmi*(ATPi_3_1)-fATPmi*(ATPm_3_1) )/ pool_ATPm ;

/*dATPm_3_2/dt=*/ out[66] = ( +fATPsm*(ATPs_3_2)+rATPmi*(ATPi_3_2)-fATPmi*(ATPm_3_2) )/ pool_ATPm ;

/*dATPm_3_3/dt=*/ out[67] = ( +fATPsm*(ATPs_3_3)+rATPmi*(ATPi_3_3)-fATPmi*(ATPm_3_3) )/ pool_ATPm ;

/*dATPo_0_0/dt=*/ out[68] = ( +fATPio*(ATPi_0_0)+fCKo*(ADPo_0*CPo_0)+rAKo*(ADPo_0*ADPo_0)-fAKo*(ATPo_0_0)-fATPoe*(ATPo_0_0)-rATPio*(ATPo_0_0)-rCKo*(ATPo_0_0) )/ pool_ATPo ;

/*dATPo_0_1/dt=*/ out[69] = ( +fATPio*(ATPi_0_1)+fCKo*(ADPo_0*CPo_1)+rAKo*(ADPo_0*ADPo_1)-fAKo*(ATPo_0_1)-fATPoe*(ATPo_0_1)-rATPio*(ATPo_0_1)-rCKo*(ATPo_0_1) )/ pool_ATPo ;

/*dATPo_0_2/dt=*/ out[70] = ( +fATPio*(ATPi_0_2)+fCKo*(ADPo_0*CPo_2)+rAKo*(ADPo_0*ADPo_2)-fAKo*(ATPo_0_2)-fATPoe*(ATPo_0_2)-rATPio*(ATPo_0_2)-rCKo*(ATPo_0_2) )/ pool_ATPo ;

/*dATPo_0_3/dt=*/ out[71] = ( +fATPio*(ATPi_0_3)+fCKo*(ADPo_0*CPo_3)+rAKo*(ADPo_0*ADPo_3)-fAKo*(ATPo_0_3)-fATPoe*(ATPo_0_3)-rATPio*(ATPo_0_3)-rCKo*(ATPo_0_3) )/ pool_ATPo ;

/*dATPo_1_0/dt=*/ out[72] = ( +fATPio*(ATPi_1_0)+fCKo*(ADPo_1*CPo_0)+rAKo*(ADPo_0*ADPo_1)-fAKo*(ATPo_1_0)-fATPoe*(ATPo_1_0)-rATPio*(ATPo_1_0)-rCKo*(ATPo_1_0) )/ pool_ATPo ;

/*dATPo_1_1/dt=*/ out[73] = ( +fATPio*(ATPi_1_1)+fCKo*(ADPo_1*CPo_1)+rAKo*(ADPo_1*ADPo_1)-fAKo*(ATPo_1_1)-fATPoe*(ATPo_1_1)-rATPio*(ATPo_1_1)-rCKo*(ATPo_1_1) )/ pool_ATPo ;

/*dATPo_1_2/dt=*/ out[74] = ( +fATPio*(ATPi_1_2)+fCKo*(ADPo_1*CPo_2)+rAKo*(ADPo_1*ADPo_2)-fAKo*(ATPo_1_2)-fATPoe*(ATPo_1_2)-rATPio*(ATPo_1_2)-rCKo*(ATPo_1_2) )/ pool_ATPo ;

/*dATPo_1_3/dt=*/ out[75] = ( +fATPio*(ATPi_1_3)+fCKo*(ADPo_1*CPo_3)+rAKo*(ADPo_1*ADPo_3)-fAKo*(ATPo_1_3)-fATPoe*(ATPo_1_3)-rATPio*(ATPo_1_3)-rCKo*(ATPo_1_3) )/ pool_ATPo ;

/*dATPo_2_0/dt=*/ out[76] = ( +fATPio*(ATPi_2_0)+fCKo*(ADPo_2*CPo_0)+rAKo*(ADPo_0*ADPo_2)-fAKo*(ATPo_2_0)-fATPoe*(ATPo_2_0)-rATPio*(ATPo_2_0)-rCKo*(ATPo_2_0) )/ pool_ATPo ;

/*dATPo_2_1/dt=*/ out[77] = ( +fATPio*(ATPi_2_1)+fCKo*(ADPo_2*CPo_1)+rAKo*(ADPo_1*ADPo_2)-fAKo*(ATPo_2_1)-fATPoe*(ATPo_2_1)-rATPio*(ATPo_2_1)-rCKo*(ATPo_2_1) )/ pool_ATPo ;

/*dATPo_2_2/dt=*/ out[78] = ( +fATPio*(ATPi_2_2)+fCKo*(ADPo_2*CPo_2)+rAKo*(ADPo_2*ADPo_2)-fAKo*(ATPo_2_2)-fATPoe*(ATPo_2_2)-rATPio*(ATPo_2_2)-rCKo*(ATPo_2_2) )/ pool_ATPo ;

/*dATPo_2_3/dt=*/ out[79] = ( +fATPio*(ATPi_2_3)+fCKo*(ADPo_2*CPo_3)+rAKo*(ADPo_2*ADPo_3)-fAKo*(ATPo_2_3)-fATPoe*(ATPo_2_3)-rATPio*(ATPo_2_3)-rCKo*(ATPo_2_3) )/ pool_ATPo ;

/*dATPo_3_0/dt=*/ out[80] = ( +fATPio*(ATPi_3_0)+fCKo*(ADPo_3*CPo_0)+rAKo*(ADPo_0*ADPo_3)-fAKo*(ATPo_3_0)-fATPoe*(ATPo_3_0)-rATPio*(ATPo_3_0)-rCKo*(ATPo_3_0) )/ pool_ATPo ;

/*dATPo_3_1/dt=*/ out[81] = ( +fATPio*(ATPi_3_1)+fCKo*(ADPo_3*CPo_1)+rAKo*(ADPo_1*ADPo_3)-fAKo*(ATPo_3_1)-fATPoe*(ATPo_3_1)-rATPio*(ATPo_3_1)-rCKo*(ATPo_3_1) )/ pool_ATPo ;

/*dATPo_3_2/dt=*/ out[82] = ( +fATPio*(ATPi_3_2)+fCKo*(ADPo_3*CPo_2)+rAKo*(ADPo_2*ADPo_3)-fAKo*(ATPo_3_2)-fATPoe*(ATPo_3_2)-rATPio*(ATPo_3_2)-rCKo*(ATPo_3_2) )/ pool_ATPo ;

/*dATPo_3_3/dt=*/ out[83] = ( +fATPio*(ATPi_3_3)+fCKo*(ADPo_3*CPo_3)+rAKo*(ADPo_3*ADPo_3)-fAKo*(ATPo_3_3)-fATPoe*(ATPo_3_3)-rATPio*(ATPo_3_3)-rCKo*(ATPo_3_3) )/ pool_ATPo ;

/*dATPs_0_0/dt=*/ out[84] = ( +fASs*((1/4.0*Ps_1+Ps_0)*ADPs_0)-fATPsm*(ATPs_0_0)-rASs*((Ws_0+Ws_1)*ATPs_0_0) )/ pool_ATPs ;

/*dATPs_0_1/dt=*/ out[85] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_1)*ADPs_0)-fATPsm*(ATPs_0_1)-rASs*((Ws_0+Ws_1)*ATPs_0_1) )/ pool_ATPs ;

/*dATPs_0_2/dt=*/ out[86] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_3)*ADPs_0)-fATPsm*(ATPs_0_2)-rASs*((Ws_0+Ws_1)*ATPs_0_2) )/ pool_ATPs ;

/*dATPs_0_3/dt=*/ out[87] = ( +fASs*((1/4.0*Ps_3+Ps_4)*ADPs_0)-fATPsm*(ATPs_0_3)-rASs*((Ws_0+Ws_1)*ATPs_0_3) )/ pool_ATPs ;

/*dATPs_1_0/dt=*/ out[88] = ( +fASs*((1/4.0*Ps_1+Ps_0)*ADPs_1)-fATPsm*(ATPs_1_0)-rASs*((Ws_0+Ws_1)*ATPs_1_0) )/ pool_ATPs ;

/*dATPs_1_1/dt=*/ out[89] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_1)*ADPs_1)-fATPsm*(ATPs_1_1)-rASs*((Ws_0+Ws_1)*ATPs_1_1) )/ pool_ATPs ;

/*dATPs_1_2/dt=*/ out[90] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_3)*ADPs_1)-fATPsm*(ATPs_1_2)-rASs*((Ws_0+Ws_1)*ATPs_1_2) )/ pool_ATPs ;

/*dATPs_1_3/dt=*/ out[91] = ( +fASs*((1/4.0*Ps_3+Ps_4)*ADPs_1)-fATPsm*(ATPs_1_3)-rASs*((Ws_0+Ws_1)*ATPs_1_3) )/ pool_ATPs ;

/*dATPs_2_0/dt=*/ out[92] = ( +fASs*((1/4.0*Ps_1+Ps_0)*ADPs_2)-fATPsm*(ATPs_2_0)-rASs*((Ws_0+Ws_1)*ATPs_2_0) )/ pool_ATPs ;

/*dATPs_2_1/dt=*/ out[93] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_1)*ADPs_2)-fATPsm*(ATPs_2_1)-rASs*((Ws_0+Ws_1)*ATPs_2_1) )/ pool_ATPs ;

/*dATPs_2_2/dt=*/ out[94] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_3)*ADPs_2)-fATPsm*(ATPs_2_2)-rASs*((Ws_0+Ws_1)*ATPs_2_2) )/ pool_ATPs ;

/*dATPs_2_3/dt=*/ out[95] = ( +fASs*((1/4.0*Ps_3+Ps_4)*ADPs_2)-fATPsm*(ATPs_2_3)-rASs*((Ws_0+Ws_1)*ATPs_2_3) )/ pool_ATPs ;

/*dATPs_3_0/dt=*/ out[96] = ( +fASs*((1/4.0*Ps_1+Ps_0)*ADPs_3)-fATPsm*(ATPs_3_0)-rASs*((Ws_0+Ws_1)*ATPs_3_0) )/ pool_ATPs ;

/*dATPs_3_1/dt=*/ out[97] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_1)*ADPs_3)-fATPsm*(ATPs_3_1)-rASs*((Ws_0+Ws_1)*ATPs_3_1) )/ pool_ATPs ;

/*dATPs_3_2/dt=*/ out[98] = ( +fASs*((1/2.0*Ps_2+3/4.0*Ps_3)*ADPs_3)-fATPsm*(ATPs_3_2)-rASs*((Ws_0+Ws_1)*ATPs_3_2) )/ pool_ATPs ;

/*dATPs_3_3/dt=*/ out[99] = ( +fASs*((1/4.0*Ps_3+Ps_4)*ADPs_3)-fATPsm*(ATPs_3_3)-rASs*((Ws_0+Ws_1)*ATPs_3_3) )/ pool_ATPs ;

/*dCPi_0/dt=*/ out[100] = ( +fCKi*(ATPi_0_0+ATPi_1_0+ATPi_2_0+ATPi_3_0)+rCio*(CPo_0)-fCio*(CPi_0)-rCKi*((ADPi_0+ADPi_1+ADPi_2+ADPi_3)*CPi_0) )/ pool_CPi ;

/*dCPi_1/dt=*/ out[101] = ( +fCKi*(ATPi_0_1+ATPi_1_1+ATPi_2_1+ATPi_3_1)+rCio*(CPo_1)-fCio*(CPi_1)-rCKi*((ADPi_0+ADPi_1+ADPi_2+ADPi_3)*CPi_1) )/ pool_CPi ;

/*dCPi_2/dt=*/ out[102] = ( +fCKi*(ATPi_0_2+ATPi_1_2+ATPi_2_2+ATPi_3_2)+rCio*(CPo_2)-fCio*(CPi_2)-rCKi*((ADPi_0+ADPi_1+ADPi_2+ADPi_3)*CPi_2) )/ pool_CPi ;

/*dCPi_3/dt=*/ out[103] = ( +fCKi*(ATPi_0_3+ATPi_1_3+ATPi_2_3+ATPi_3_3)+rCio*(CPo_3)-fCio*(CPi_3)-rCKi*((ADPi_0+ADPi_1+ADPi_2+ADPi_3)*CPi_3) )/ pool_CPi ;

/*dCPo_0/dt=*/ out[104] = ( +fCio*(CPi_0)+rCKo*(ATPo_0_0+ATPo_1_0+ATPo_2_0+ATPo_3_0)-fCKo*((ADPo_0+ADPo_1+ADPo_2+ADPo_3)*CPo_0)-rCio*(CPo_0) )/ pool_CPo ;

/*dCPo_1/dt=*/ out[105] = ( +fCio*(CPi_1)+rCKo*(ATPo_0_1+ATPo_1_1+ATPo_2_1+ATPo_3_1)-fCKo*((ADPo_0+ADPo_1+ADPo_2+ADPo_3)*CPo_1)-rCio*(CPo_1) )/ pool_CPo ;

/*dCPo_2/dt=*/ out[106] = ( +fCio*(CPi_2)+rCKo*(ATPo_0_2+ATPo_1_2+ATPo_2_2+ATPo_3_2)-fCKo*((ADPo_0+ADPo_1+ADPo_2+ADPo_3)*CPo_2)-rCio*(CPo_2) )/ pool_CPo ;

/*dCPo_3/dt=*/ out[107] = ( +fCio*(CPi_3)+rCKo*(ATPo_0_3+ATPo_1_3+ATPo_2_3+ATPo_3_3)-fCKo*((ADPo_0+ADPo_1+ADPo_2+ADPo_3)*CPo_3)-rCio*(CPo_3) )/ pool_CPo ;

/*dPe_0/dt=*/ out[108] = ( +fASe*((ATPe_0_0+ATPe_1_0+ATPe_2_0+ATPe_3_0)*We_0)+rPeo*(Po_0)-fPeo*(Pe_0)-rASe*((ADPe_0+ADPe_1+ADPe_2+ADPe_3)*Pe_0) )/ pool_Pe ;

/*dPe_1/dt=*/ out[109] = ( +fASe*((ATPe_0_0+ATPe_1_0+ATPe_2_0+ATPe_3_0)*We_1+(ATPe_0_1+ATPe_1_1+ATPe_2_1+ATPe_3_1)*We_0)+rPeo*(Po_1)-fPeo*(Pe_1)-rASe*((ADPe_0+ADPe_1+ADPe_2+ADPe_3)*Pe_1) )/ pool_Pe ;

/*dPe_2/dt=*/ out[110] = ( +fASe*((ATPe_0_1+ATPe_1_1+ATPe_2_1+ATPe_3_1)*We_1+(ATPe_0_2+ATPe_1_2+ATPe_2_2+ATPe_3_2)*We_0)+rPeo*(Po_2)-fPeo*(Pe_2)-rASe*((ADPe_0+ADPe_1+ADPe_2+ADPe_3)*Pe_2) )/ pool_Pe ;

/*dPe_3/dt=*/ out[111] = ( +fASe*((ATPe_0_2+ATPe_1_2+ATPe_2_2+ATPe_3_2)*We_1+(ATPe_0_3+ATPe_1_3+ATPe_2_3+ATPe_3_3)*We_0)+rPeo*(Po_3)-fPeo*(Pe_3)-rASe*((ADPe_0+ADPe_1+ADPe_2+ADPe_3)*Pe_3) )/ pool_Pe ;

/*dPe_4/dt=*/ out[112] = ( +fASe*((ATPe_0_3+ATPe_1_3+ATPe_2_3+ATPe_3_3)*We_1)+rPeo*(Po_4)-fPeo*(Pe_4)-rASe*((ADPe_0+ADPe_1+ADPe_2+ADPe_3)*Pe_4) )/ pool_Pe ;

/*dPm_0/dt=*/ out[113] = ( +fPom*(Po_0)+rPms*(Ps_0)-fPms*(Pm_0) )/ pool_Pm ;

/*dPm_1/dt=*/ out[114] = ( +fPom*(Po_1)+rPms*(Ps_1)-fPms*(Pm_1) )/ pool_Pm ;

/*dPm_2/dt=*/ out[115] = ( +fPom*(Po_2)+rPms*(Ps_2)-fPms*(Pm_2) )/ pool_Pm ;

/*dPm_3/dt=*/ out[116] = ( +fPom*(Po_3)+rPms*(Ps_3)-fPms*(Pm_3) )/ pool_Pm ;

/*dPm_4/dt=*/ out[117] = ( +fPom*(Po_4)+rPms*(Ps_4)-fPms*(Pm_4) )/ pool_Pm ;

/*dPo_0/dt=*/ out[118] = ( +fPeo*(Pe_0)-fPom*(Po_0)-rPeo*(Po_0) )/ pool_Po ;

/*dPo_1/dt=*/ out[119] = ( +fPeo*(Pe_1)-fPom*(Po_1)-rPeo*(Po_1) )/ pool_Po ;

/*dPo_2/dt=*/ out[120] = ( +fPeo*(Pe_2)-fPom*(Po_2)-rPeo*(Po_2) )/ pool_Po ;

/*dPo_3/dt=*/ out[121] = ( +fPeo*(Pe_3)-fPom*(Po_3)-rPeo*(Po_3) )/ pool_Po ;

/*dPo_4/dt=*/ out[122] = ( +fPeo*(Pe_4)-fPom*(Po_4)-rPeo*(Po_4) )/ pool_Po ;

/*dPs_0/dt=*/ out[123] = ( +fPms*(Pm_0)+rASs*((ATPs_0_0+ATPs_1_0+ATPs_2_0+ATPs_3_0)*Ws_0)-fASs*((ADPs_0+ADPs_1+ADPs_2+ADPs_3)*Ps_0)-rPms*(Ps_0) )/ pool_Ps ;

/*dPs_1/dt=*/ out[124] = ( +fPms*(Pm_1)+rASs*((ATPs_0_0+ATPs_1_0+ATPs_2_0+ATPs_3_0)*Ws_1+(ATPs_0_1+ATPs_1_1+ATPs_2_1+ATPs_3_1)*Ws_0)-fASs*((ADPs_0+ADPs_1+ADPs_2+ADPs_3)*Ps_1)-rPms*(Ps_1) )/ pool_Ps ;

/*dPs_2/dt=*/ out[125] = ( +fPms*(Pm_2)+rASs*((ATPs_0_1+ATPs_1_1+ATPs_2_1+ATPs_3_1)*Ws_1+(ATPs_0_2+ATPs_1_2+ATPs_2_2+ATPs_3_2)*Ws_0)-fASs*((ADPs_0+ADPs_1+ADPs_2+ADPs_3)*Ps_2)-rPms*(Ps_2) )/ pool_Ps ;

/*dPs_3/dt=*/ out[126] = ( +fPms*(Pm_3)+rASs*((ATPs_0_2+ATPs_1_2+ATPs_2_2+ATPs_3_2)*Ws_1+(ATPs_0_3+ATPs_1_3+ATPs_2_3+ATPs_3_3)*Ws_0)-fASs*((ADPs_0+ADPs_1+ADPs_2+ADPs_3)*Ps_3)-rPms*(Ps_3) )/ pool_Ps ;

/*dPs_4/dt=*/ out[127] = ( +fPms*(Pm_4)+rASs*((ATPs_0_3+ATPs_1_3+ATPs_2_3+ATPs_3_3)*Ws_1)-fASs*((ADPs_0+ADPs_1+ADPs_2+ADPs_3)*Ps_4)-rPms*(Ps_4) )/ pool_Ps ;

/*dWe_0/dt=*/ out[128] = ( +rASe*((1/2.0*Pe_2+1/4.0*Pe_3+3/4.0*Pe_1+Pe_0)*(ADPe_0+ADPe_1+ADPe_2+ADPe_3))+rWeo*(Wo_0)-fASe*((ATPe_0_0+ATPe_0_1+ATPe_0_2+ATPe_0_3+ATPe_1_0+ATPe_1_1+ATPe_1_2+ATPe_1_3+ATPe_2_0+ATPe_2_1+ATPe_2_2+ATPe_2_3+ATPe_3_0+ATPe_3_1+ATPe_3_2+ATPe_3_3)*We_0)-fWeo*(We_0) )/ pool_We ;

/*dWe_1/dt=*/ out[129] = ( +rASe*((1/2.0*Pe_2+1/4.0*Pe_1+3/4.0*Pe_3+Pe_4)*(ADPe_0+ADPe_1+ADPe_2+ADPe_3))+rWeo*(Wo_1)-fASe*((ATPe_0_0+ATPe_0_1+ATPe_0_2+ATPe_0_3+ATPe_1_0+ATPe_1_1+ATPe_1_2+ATPe_1_3+ATPe_2_0+ATPe_2_1+ATPe_2_2+ATPe_2_3+ATPe_3_0+ATPe_3_1+ATPe_3_2+ATPe_3_3)*We_1)-fWeo*(We_1) )/ pool_We ;

/*dWs_0/dt=*/ out[130] = ( +fASs*((1/2.0*Ps_2+1/4.0*Ps_3+3/4.0*Ps_1+Ps_0)*(ADPs_0+ADPs_1+ADPs_2+ADPs_3))+fWos*(Wo_0)-rASs*((ATPs_0_0+ATPs_0_1+ATPs_0_2+ATPs_0_3+ATPs_1_0+ATPs_1_1+ATPs_1_2+ATPs_1_3+ATPs_2_0+ATPs_2_1+ATPs_2_2+ATPs_2_3+ATPs_3_0+ATPs_3_1+ATPs_3_2+ATPs_3_3)*Ws_0)-rWos*(Ws_0) )/ pool_Ws ;

/*dWs_1/dt=*/ out[131] = ( +fASs*((1/2.0*Ps_2+1/4.0*Ps_1+3/4.0*Ps_3+Ps_4)*(ADPs_0+ADPs_1+ADPs_2+ADPs_3))+fWos*(Wo_1)-rASs*((ATPs_0_0+ATPs_0_1+ATPs_0_2+ATPs_0_3+ATPs_1_0+ATPs_1_1+ATPs_1_2+ATPs_1_3+ATPs_2_0+ATPs_2_1+ATPs_2_2+ATPs_2_3+ATPs_3_0+ATPs_3_1+ATPs_3_2+ATPs_3_3)*Ws_1)-rWos*(Ws_1) )/ pool_Ws ;

} 
