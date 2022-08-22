import numpy as np
from collections import OrderedDict

from differentiator import differentiate
from exporttree import python_code
from tree import NODETYPE

# d:            Differentiator
# h:            x value difference used for the finite differences approximation
# err_limit:    Maximum allowed absolute error
# verbose:      Output lots of intermediate information or not
def numcheck(originalDag, diffDag, variable_ranks, arg_name, h=1e-8, err_limit=1e-6, verbose=False):
    if verbose: 
        print('Numerical check')
        print('-----------------')
        print(f'Original expression:   {originalDag}')
        print(f'Derivative expression: {diffDag}')
    axis_length = 3 # We set all axes to this length
    variable_names = list(variable_ranks.keys())
    (deltas, ranks) = get_deltas_in_input(originalDag)
    variable_names_string = ','.join(variable_names + deltas)
    f = eval(f'lambda {variable_names_string}: {python_code(originalDag)}') # Create a function that computes the expression represented by the dag
    df = eval(f'lambda {variable_names_string}: {python_code(diffDag)}')
    if verbose:
        print(f'Code generated for original expression:   {python_code(originalDag)}')
        print(f'Code generated for derivative expression: {python_code(diffDag)}')

    variables = OrderedDict()
    for variable_name in variable_names:
        shape = ()
        for axis in range(variable_ranks[variable_name]):
            shape += (axis_length,) # Construct a tuple where each entry is the length of the corresponding axis, always axis_length
        variables[variable_name] = np.random.random_sample(shape) # All variables used in the expression with fitting shapes and random values. All axes have length axis_length
    for delta in deltas:
        shape = ()
        for axis in range(ranks[delta]):
            shape += (axis_length,)
        variables[delta] = np.random.random_sample(shape)
    approx_shape = () # Shape of the output of df
    for axis in range(diffDag.rank):
        approx_shape += (axis_length,)
    df_approx = np.zeros(approx_shape)

    X = variables[arg_name]
    it = np.nditer(X, flags=['multi_index']) # Iterator with a multi-index over all elements in the argument
    for entry in it: #  One iteration = Approximate the derivative of f with respect to one entry of X
        original_value = X[it.multi_index]
        X[it.multi_index] = original_value + h # Add h to the current entry
        f_x_plus_h = f(*variables.values()) # the * operator unpacks the list into separate function arguments
        X[it.multi_index] = original_value - h
        f_x_minus_h = f(*variables.values())
        X[it.multi_index] = original_value
        df_approx[(...,) + it.multi_index] = (f_x_plus_h - f_x_minus_h) / (2*h)
    df_computed = df(*variables.values())
    abs_error = (df_approx - df_computed)
    check_passed = np.all(abs_error < err_limit)
    if verbose:
        print(f'Approximate derivative value: {df_approx}')
        print(f'Computed derivative value: {df_computed}')
        print(f'Absolute Error: {abs_error}')
        if not check_passed: print('Check failed')
        else: print('Check passed')
    return check_passed

def get_deltas_in_input(originalDag):
    deltas = []
    ranks = {}
    def helper(node):
        if not node: return
        if node.type == NODETYPE.DELTA and node.name not in deltas:
            deltas.append(node.name)
            ranks[node.name] = node.rank
        helper(node.left)
        helper(node.right)
    helper(originalDag)
    return deltas, ranks