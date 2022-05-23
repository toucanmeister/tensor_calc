from numpy import diff
from differentiator import Differentiator

def example_broadcasting():
    example= '''
        declare
          A 2
          x 1
        expression (A + 1) *(ij,j->i) x
        derivative wrt x
    '''
    differentiator = Differentiator(example)
    differentiator.differentiate()
    differentiator.originalDag.dot('example_broadcasting/example_broadcastingOriginal')
    differentiator.diffDag.dot('example_broadcasting/example_broadcastingDiff')
    print(f'example_broadcasting: {differentiator.diffDag}')

def example_power_1():
    example= '''
        declare
          x 2
        expression x^3
        derivative wrt x
    '''
    differentiator = Differentiator(example)
    differentiator.differentiate()
    differentiator.originalDag.dot('example_power_1/example_power_1Original')
    differentiator.diffDag.dot('example_power_1/example_power_1Diff')
    print(f'example_power_1: {differentiator.diffDag}')

def example_power_2():
    example= '''
        declare
          x 1
          y 1
        expression y^(x *(i,i->) y)
        derivative wrt x
    '''
    differentiator = Differentiator(example)
    differentiator.differentiate()
    differentiator.originalDag.dot('example_power_2/example_power_2Original')
    differentiator.diffDag.dot('example_power_2/example_power_2Diff')
    print(f'example_power_2: {differentiator.diffDag}')

def example_inv_det():
    example= '''
        declare
          X 2
        expression det(X) *(,ij->ij) inv(X) 
        derivative wrt X
    '''
    differentiator = Differentiator(example)
    differentiator.differentiate()
    differentiator.originalDag.dot('example_inv_det/example_inv_detOriginal')
    differentiator.diffDag.dot('example_inv_det/example_inv_detDiff')
    print(f'example_inv_det: {differentiator.diffDag}')


if __name__ == '__main__':
    #example_broadcasting()
    #example_power_1()
    #example_power_2()
    example_inv_det()