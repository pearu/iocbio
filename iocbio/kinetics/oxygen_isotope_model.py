'''
Definition of OxygenIsotopeModel by abstraction from an IsotopeModel class.

To use the model builder, one must:
  Define a function check_reaction:
    Return True when reaction is possible for given tuples of
    reactant and product indices. Reaction pattern is a string in
    a form 'R1+R2(<-|->)P1-P2' where reactants R1, R2 and products
    P1, P2 are keys of the species dictionary.

  Define an index_dic which contains the definition of the
  oxygen isotopologues for each species in the model:
    For the case of one ADP species, the index_dic is defined as:
    index_dic = dict(ADPo = ['000', '001', '010', '011', '100', '101', '110', '111'])
    Where '0' and '1' refer to labeled and unlabeled oxygen atoms.

  Run this script with a python interpreter:
  python oxygen_isotope_model.py

  Two files are generated.  The first contains the model and the
  second contains the variables used.

'''

from __future__ import division

import itertools

from builder import IsotopeModel

oxygen_isotope_system_str = '''

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

AKi  : ADPi + ADPi <=> ATPi
AKo  : ATPo <=> ADPo + ADPo

CKi   : ATPi <=> ADPi + CPi
CKo   : CPo + ADPo <=> ATPo 

ADPim : ADPi <=> ADPm
ADPoi : ADPo <=> ADPi 

ATPmi : ATPm <=> ATPi
ATPio : ATPi <=> ATPo

Cio   : CPi <=> CPo
Pom   : Po => Pm
'''


# The following code section makes creating the index dic easier. 
make_indices = lambda repeat: map(''.join,itertools.product('01', repeat=repeat)) if repeat else ['']

n = 3
T1indices = make_indices(n)
T2indices = make_indices(n)
W_indices = make_indices(1)
ADP_indices = make_indices(n)
CP_indices = make_indices(n)
P_indices = make_indices(n+1)

ATP_indices = []
for t1 in T1indices:
    for t2 in T2indices:
        ATP_indices.append (t1+'_'+t2)
        
class OxygenIsotopeModel(IsotopeModel):

    index_dic = dict(ATPm=ATP_indices,
                     ADPm=ADP_indices,
                     Pm=P_indices,
                     ATPi=ATP_indices,
                     ADPi=ADP_indices,
                     CPi=CP_indices,
                     ATPo=ATP_indices,
                     Wo=W_indices,
                     ADPo=ADP_indices,
                     Po=P_indices,
                     CPo=CP_indices,
                     ATPe=ATP_indices,
                     We=W_indices,
                     ADPe=ADP_indices,
                     Pe=P_indices,
                     ATPs=ATP_indices,
                     Ws=W_indices,
                     ADPs=ADP_indices,
                     Ps=P_indices,
                     )

    # The current interface requires defining check_reaction for all reactions.
    # The following code defines all of the transport reactions. 
    transport_reactions = []
    for a, b in [('ATPi', 'ATPo'), ('ATPm', 'ATPi'),
                 ('ADPo', 'ADPi'), ('ADPi', 'ADPm'), ('ADPm', 'ADPs'), ('ADPe', 'ADPo'),
                 ('Pm', 'Ps'), ('Pe', 'Po'),
                 ('Wo', 'Ws'), ('We', 'Wo'),
                 ('CPi', 'CPo'),        
                 ]:
        transport_reactions.append('%s->%s' % (a,b))
        transport_reactions.append('%s<-%s' % (a,b))
        
    for a, b in [('Po', 'Pm'), ('ATPo', 'ATPe'), ('ATPs', 'ATPm'),]:
        transport_reactions.append('%s->%s' % (a,b))

    def check_reaction(self, reaction_pattern, rindices, pindices):
        if reaction_pattern in self.transport_reactions:
            return rindices == pindices

        # All of the enzymatic reactions in the model require their
        # own definitions for the atom mappings. The builder splits
        # ATP into two indices because each phosphoryl group is treated
        # separately in the symbolic calculation to aid in simplifying
        # the system. 
        if reaction_pattern in ['CPo+ADPo->ATPo', 'CPo+ADPo<-ATPo']:
            atp, = pindices
            cp, adp = rindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == cp
        if reaction_pattern in ['ATPi->ADPi+CPi', 'ATPi<-ADPi+CPi']:
            atp, = rindices
            adp, cp = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == cp
        if reaction_pattern in ['ADPi+ADPi->ATPi','ADPi+ADPi<-ATPi']:
            adp, adp2 = rindices
            atp, = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == adp2
        if reaction_pattern in ['ATPo->ADPo+ADPo','ATPo<-ADPo+ADPo']:
            atp, = rindices
            adp, adp2 = pindices
            t1, t2 = atp.split('_')
            return t1 == adp and t2 == adp2
        if reaction_pattern in ['ATPe+We->ADPe+Pe', 'ATPe+We<-ADPe+Pe']:
            atp, w = rindices
            adp, p = pindices
            t1, t2 = atp.split ('_')
            return t1==adp and (t2+w).count('1')==p.count ('1')
        if reaction_pattern in ['ADPs+Ps->ATPs+Ws', 'ADPs+Ps<-ATPs+Ws']:
            atp, w = pindices
            adp, p = rindices
            t1, t2 = atp.split ('_')
            return t1==adp and (t2+w).count('1')==p.count ('1')

        # If a reaction is not implemented, it will return False: 
        return IsotopeModel.check_reaction(self, reaction_pattern, rindices, pindices)
    
if __name__ == '__main__':
            
    frac_W = 0.3
    water_labeling = {'0':1 - frac_W,
                      '1':frac_W}
                      
    model = OxygenIsotopeModel(system_string=oxygen_isotope_system_str,
                               water_labeling=water_labeling)
    
    model.compile_ccode(debug=False, stage=None)

