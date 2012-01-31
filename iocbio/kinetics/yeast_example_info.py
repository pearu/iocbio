import os

sbml_file = 'yeast_example'

if not os.path.isfile (sbml_file + '.xml'):
    raise RuntimeError ('Failed to load SBML XML file for yeast_example')

external_fluxes = dict(R_GL_in=100,
                       R_FviP_out=11,
                       R_GviP_out=3.8,
                       R_GiiiP_out=0.45,
                       R_RvP_out=2.6,
                       R_OA_out=0.36,
                       R_AcCoA_out=0.3,
                       R_ASP_out=2.39,
                       R_ARG_out=1.94,
                       R_GLY_out=1.17,
                       R_MET_out=0.51,
                       R_THR_out=1.54,
                       R_ILE_out=2.33,
                       R_ASN_out=0.82,
                       R_ALA_out=2.77,
                       R_GLU_out=3.04,
                       R_GLN_out=1.06,
                       R_HIS_out=0.8,
                       R_LEU_out=3.57,
                       R_LYS_out=3.45,
                       R_PHE_out=2.43,
                       R_PRO_out=1.66,
                       R_SER_out=1.12,
                       R_TRP_out=0.62,
                       R_TYR_out=1.84,
                       R_VAL_out=2.66)

internal_fluxes = dict(R_IOSm_IOSo=0,
                       R_GviP_RLvP=5,
                       R_TCA_MAm_PYm=0,
                       R_OA_PEP=0,
                       R_B_THR_GLY=0,
                       R_B_PY_AcCoA_LEU_M=2,
                       R_B_PY_VAL_M=1,
                       R_ASPm_ASPo=2,
                       R_PYo_AAo=10,
                       R_B_GLT=4,
                       R_B_PY_ALA_M1=1.0,
                       R_US_C_ASP_AS=2.5,
                       R_OGo_OGm=0,
                       )

internal_fluxes_A = dict(R_IOSm_IOSo=0,
                       R_PPP_FviP_EivP_SviiP_GiiiP_01=1.958,
                       R_TCA_MAm_PYm=0,
                       R_OA_PEP=0,
                       R_B_THR_GLY=0,
                       R_B_PY_AcCoA_LEU_M=2,
                       R_B_PY_VAL_M=1,
                       R_ASPm_ASPo=2,
                       R_PYo_AAo=10,
                       R_B_GLT=4,
                       R_B_PY_ALA_M1=1.0,
                       R_US_C_ASP_AS=2.5,
                       R_OGo_OGm=0,
                       )

