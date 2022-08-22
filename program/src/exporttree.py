#author: julien klaus
#email: julien.klaus@uni-jena.de
#data: 01.06.2022

import re

from tree import NODETYPE


op_to_np = {
    '+': 'np.add', 
    '-': 'np.subtract',
    '^': 'np.power',
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'arcsin': 'np.arcsin',
    'arccos': 'np.arccos',
    'arctan': 'np.arctan',
    'tanh': 'np.tanh',
    'exp': 'np.exp', 
    'log': 'np.log',
    'sign': 'np.sign',
    'relu': '(lambda x: np.maximum(x,0))',
    'abs': 'np.abs',
    'det': 'np.linalg.det',
    'inv': 'np.linalg.inv',
    'adj': '(lambda x: np.multiply(np.linalg.det(x), np.linalg.inv(x)))'
}

def _get_missing_axis(dag):
    axis_to_numpy = {}
    for axis, origin in dag.companion.axis_to_origin.items():
        var_name = origin.split('[')[0]
        index = origin.split('[')[1].split(']')[0]
        axis_to_numpy[axis] = f'np.shape({var_name})[{index}]'
    for node in dag.get_all_subtrees():
        if node and node.type == NODETYPE.VARIABLE:
            for index, axis in enumerate(node.axes):
               if not axis in axis_to_numpy:
                    axis_to_numpy[axis] = f'np.shape({node.name})[{index}]'
    return axis_to_numpy


def python_code(dag):
    axis_to_numpy = _get_missing_axis(dag)
    return _code(dag, axis_to_numpy)


def _get_shape_from_axis(node, axes):
    shape = []
    for dim in node.axes:
        if not dim in axes:
            raise Exception(f'No information for dimension of node {node.name} (axis {dim})')
        else:
            shape.append(axes[dim])
    return shape


def _delta(delta_node, axes):
    shape = _get_shape_from_axis(delta_node, axes)
    eye_len = '1.0'
    for i in range(int(delta_node.rank/2)):
        eye_len = f'{eye_len}*{shape[i]}'
    if shape == []:
        delta_string = f'np.eye(int({eye_len}),int({eye_len})).squeeze()'
    else:
        delta_string = f'np.eye(int({eye_len}),int({eye_len})).reshape({",".join(shape)})'
    return delta_string


def _code(t, axes):
    if t.type == NODETYPE.DELTA:
        return _delta(t, axes)

    elif t.name in ['+', '-', '^']:
        if t.left:
            return f'{op_to_np[t.name]}({_code(t.left, axes)},{_code(t.right, axes)})'
        else:
            return f'{t.name}({_code(t.right, axes)})'

    elif t.type in [NODETYPE.ELEMENTWISE_FUNCTION, NODETYPE.SPECIAL_FUNCTION]: # unary operations store their argument in the right tree
        if t.name in op_to_np:
            return f'{op_to_np[t.name]}({_code(t.right, axes)})'
        elif t.name == 'elementwise_inverse':
            return f'np.divide(1,{_code(t.right, axes)})'
        else:
            raise NotImplementedError(f'Operation {t.name} is not implemented yet.')

    elif t.type == NODETYPE.VARIABLE:
        return t.name
    elif t.type == NODETYPE.CONSTANT:
        shape = _get_shape_from_axis(t, axes)
        constant_val = t.name.split('_')[0]
        return f'np.full(({",".join(shape)}),{constant_val})'

    elif t.type == NODETYPE.PRODUCT:
        einsum_string = f'{t.leftIndices},{t.rightIndices}->{t.resultIndices}'
        return f'np.einsum(\'{einsum_string}\',{_code(t.left, axes)},{_code(t.right, axes)})'

    else:
        raise NotImplementedError(f'Operation {t.name} is not implemented yet.')

