"""

See
  http://code.google.com/p/iocbio/wiki/OxygenIsotopeEquationGenerator
  

"""

from builder import IsotopeModel

class DemoModel(IsotopeModel):
    system_string = '''
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
        if l.match('Ai => A', transport=True): return l.Ai==l.A
        if l.match('C => Co', transport=True): return l.C==l.Co
        if l.match('A+B <=> C'): 
            t1,t2 = l.C.split ('_')
            return l.A==t1 and l.B==t2
        # Unknown reaction, raise an exception
        return IsotopeModel.check_reaction(self, reaction)


if __name__ == '__main__':

    # Create a model instance.
    model = DemoModel()
    
    # Demonstrate model equations
    model.demo()

    # Generate C code with kinetic equations
    model.compile_ccode(debug=False)
