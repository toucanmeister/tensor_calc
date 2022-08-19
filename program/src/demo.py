from differentiator import Differentiator
from numcheck import numcheck

if __name__ == '__main__':
    example = '''
        declare 
            v 1 
        expression (delta(1) + 1) *(ij,i->i) (v *(i,i->i) v) 
        derivative wrt v
    '''
    d = Differentiator(example)
    d.differentiate()
    d.render()
    d2 = Differentiator(f'declare v 1 expression {d.diffDag} derivative wrt v')
    d2.differentiate()
    numcheck(d2, verbose=True)