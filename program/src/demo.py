from differentiator import Differentiator
from numcheck import numcheck

if __name__ == '__main__':
    example = '''
        declare 
            A 2
        expression inv(A)
        derivative wrt A
    '''

    d = Differentiator(example)
    d.differentiate()
    numcheck(d, verbose=True)