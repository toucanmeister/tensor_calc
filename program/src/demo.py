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
    passed_checks = 0
    numcheck(d, h = 1e-7, err_limit=1e-6, verbose=True)