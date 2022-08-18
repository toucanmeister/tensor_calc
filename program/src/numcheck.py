import numpy as np
from collections import OrderedDict

from differentiator import Differentiator
from exporttree import python_code

# d:            Differentiator
# h:            x value difference used for the finite differences approximation
# err_limit:    Maximum allowed absolute error
# verbose:      Output lots of intermediate information or not
def numcheck(d, h=1e-8, err_limit=1e-6, verbose=False):
    if verbose: 
        print('Numerical check')
        print('-----------------')
        print(f'Original expression:   {d.originalDag}')
        print(f'Derivative expression: {d.diffDag}')
    axis_length = 3 # We set all axes to this length
    variable_names = d.variable_ranks.keys()
    variable_names_string = ','.join(variable_names)
    f = eval(f'lambda {variable_names_string}: {python_code(d.originalDag)}') # Create a function that computes the expression represented by the dag
    df = eval(f'lambda {variable_names_string}: {python_code(d.diffDag)}')
    if verbose:
        print(f'Code generated for original expression:   {python_code(d.originalDag)}')
        print(f'Code generated for derivative expression: {python_code(d.diffDag)}')

    variables = OrderedDict()
    for variable_name in variable_names:
        shape = ()
        for axis in range(d.variable_ranks[variable_name]):
            shape += (axis_length,) # Construct a tuple where each entry is the length of the corresponding axis, always axis_length
        variables[variable_name] = np.random.random_sample(shape) # All variables used in the expression with fitting shapes and random values. All axes have length axis_length
    approx_shape = () # Shape of the output of df
    for axis in range(d.diffDag.rank):
        approx_shape += (axis_length,)
    df_approx = np.zeros(approx_shape)

    X = variables[d.arg.name]
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