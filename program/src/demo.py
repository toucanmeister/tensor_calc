from numpy import diff
from differentiator import Differentiator

if __name__ == '__main__':
    example= '''
        declare
            a 0
            X 2
            Y 2
            W 2
        expression #Insert Expression Here
        derivative wrt W
    '''
    differentiator = Differentiator(example)
    differentiator.differentiate()
    differentiator.diffDag.dot('example_broadcasting/example_broadcastingDiff')
    