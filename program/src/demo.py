from differentiator import differentiate
from parser import parse
from numcheck import numcheck

if __name__ == '__main__':
    example= '''
        declare X 2 
        expression 1*(,ij->)X 
        derivative wrt X
    '''
    diffDag, originalDag, arg_name, variable_ranks = differentiate(example)
    numcheck(originalDag, diffDag, variable_ranks, arg_name, verbose=True)
