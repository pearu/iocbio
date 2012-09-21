'''
Definition of OxygenIsotopeModel by abstraction from an IsotopeModel class.

To use the model builder, one must derive a class from IsotopeModel.
Derived class must define:
  * system_string attribute that contains the definitions of reactions
    and labeling information of species;
  * check_reaction(reaction) method that defines the valid atom mappings
    of species for each reaction.

Next, run this script with a python interpreter:
  python oxygen_isotope_model.py

Two files are generated.  The first contains the model as C source code and the
second contains the variables used.

This example script corresponds to the system analyzed in

  David W. Schryer, Pearu Peterson, Ardo Illaste, Marko Vendelin.
  Sensitivity analysis of flux determination in heart by
  H218O-provided labeling using a dynamic isotopologue model of energy
  transfer pathways.
  PLoS Computational Biology.

For more information, see

  http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator
'''

from builder import IsotopeModel

class OxygenIsotopeModel(IsotopeModel):
    system_string = '''
# See IsotopeModel.parse_system_string.__doc__ for syntax of the system string.
#
# Definitions of specie labeling
# <specie name> = <specie name>[<labeling pattern>]
#
ADPs+ADPm+ADPi+ADPo+ADPe = ADPs[***]+ADPm[***]+ADPi[***]+ADPo[***]+ADPe[***]
Ps+Pm+Pe+Po = Ps[****]+Pm[****]+Pe[****]+Po[****]
Wo+Ws+We = Wo[*]+Ws[*]+We[*]
CPi+CPo = CPi[***]+CPo[***]
ATPs+ATPm+ATPo+ATPe+ATPi = ATPs[***_***]+ATPm[***_***]+ATPo[***_***]+ATPe[***_***]+ATPi[***_***]

#
# Definitions of reactions
# <reaction name> : <sum of reactants> (<=|=>|<=>) <sum of products>
#
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
'''

    def check_reaction(self, reaction):
        """ Validate reaction.

        For documentation, read the comments below. Also see
        IsotopeModel.check_reaction.__doc__ for more details.
        """
        # Create label matcher object
        l = self.labels(reaction)
        # Note that the label matcher object is special Python
        # dictionary object that items can be accessed via attributes.
        # Keys are the names of species and values are labeling
        # indices.

        # First, handle transport reactions. Usage of `transport=True`
        # improves performance but, in general, it is not required.
        if l.match('ADPm <=> ADPs', transport=True): return l.ADPm==l.ADPs
        if l.match('Pm <=> Ps', transport=True): return l.Pm==l.Ps
        if l.match('Wo <=> Ws', transport=True): return l.Wo==l.Ws
        if l.match('ATPs => ATPm', transport=True): return l.ATPs==l.ATPm
        if l.match('ATPo => ATPe', transport=True): return l.ATPo==l.ATPe
        if l.match('Pe <=> Po', transport=True): return l.Pe==l.Po
        if l.match('We <=> Wo', transport=True): return l.We==l.Wo
        if l.match('ADPe <=> ADPo', transport=True): return l.ADPe==l.ADPo
        if l.match('ADPi <=> ADPm', transport=True): return l.ADPi==l.ADPm
        if l.match('ADPo <=> ADPi', transport=True): return l.ADPo==l.ADPi
        if l.match('ATPm <=> ATPi', transport=True): return l.ATPm==l.ATPi
        if l.match('ATPi <=> ATPo', transport=True): return l.ATPi==l.ATPo
        if l.match('CPi <=> CPo', transport=True): return l.CPi==l.CPo
        if l.match('Po => Pm', transport=True): return l.Po==l.Pm

        # All of the enzymatic reactions in the model require their
        # own definitions for the atom mappings. The builder splits
        # ATP into two indices because each phosphoryl group is treated
        # separately in the symbolic calculation to aid in simplifying
        # the system.
        if l.match('Ps + ADPs <=> ATPs + Ws'):
            t1, t2 = l.ATPs.split ('_')
            return t1==l.ADPs and (t2+l.Ws).count('1')==l.Ps.count ('1')
        if l.match('Pe + ADPe <=> ATPe + We'):
            t1, t2 = l.ATPe.split ('_')
            return t1==l.ADPe and (t2+l.We).count('1')==l.Pe.count ('1')
        if l.match('CPo + ADPo <=> ATPo'):
            t1, t2 = l.ATPo.split ('_')
            return t1 == l.ADPo and t2==l.CPo
        if l.match('CPi + ADPi <=> ATPi'):
            t1, t2 = l.ATPi.split ('_')
            return t1 == l.ADPi and t2==l.CPi
        # notice that '2 A <=> B' has to be written as 'A + A <=> B'
        if l.match('ADPi + ADPi <=> ATPi'): 
            t1, t2 = l.ATPi.split ('_')
            return t1==l.ADPi[0] and t2==l.ADPi[1]
        if l.match('ADPo + ADPo <=> ATPo'):
            t1, t2 = l.ATPo.split ('_')
            return t1==l.ADPo[0] and t2==l.ADPo[1]

        # Unknown reaction, raise an exception
        return IsotopeModel.check_reaction(self, reaction)
    
if __name__ == '__main__':

    # Specify species with defined labeling states.  Keys are the
    # names of isotope species and values are expressions defining the
    # labeling states of the corresponding species.
    defined_labeling = dict(Wo_0='0.7', Wo_1='0.3')

    # Create a model instance.
    model = OxygenIsotopeModel(defined_labeling=defined_labeling,
                               model_name='model_3000')

    # Generate C source code of the model.
    model.compile_ccode(debug=False, stage=None)

