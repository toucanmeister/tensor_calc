from scanner import ELEMENTWISE_FUNCTIONS, TOKEN_ID
from tree import Tree, NODETYPE, TreeCompanion
from parser import parse
import string

# This module contains a method to differentiate an expression DAG with respect to an argument

# Note:
# -----------------------------------------------------------------
# Since trees are recursive data structures, we store information
# about them in companion objects. Since a lot of information from
# the input DAG is used by the differentation DAG, both of these
# have the same companion object.
# -----------------------------------------------------------------

_originalNodeToDiffNode = {} # For a node X, we save where dY/dX is, to allow adding more chain rule contributions when we reach that node again later
_originalNodeToDiffTree = {} # For a node X, we also save the top of the tree which contains dX/dZ (for possibly 2 nodes Z), which also we need to update when adding chain rule contributions

def differentiate(input):
    global _originalNodeToDiffNode
    _originalNodeToDiffNode = {}   
    global _originalNodeToDiffTree
    _originalNodeToDiffTree = {}

    originalDag, arg_name, variable_ranks = parse(input)
    originalDag, arg = _preprocess(originalDag, arg_name, variable_ranks)

    originalRank = originalDag.rank
    if arg == None: # Argument is not in expression
        diffDag = Tree(NODETYPE.CONSTANT, f'0_{originalDag.companion.new_constant()}', companion=originalDag.companion)
        diffDag.rank = originalRank * 2
        diffDag.axes = originalDag.axes + originalDag.axes
        return diffDag, originalDag, arg_name, variable_ranks
    diffDag = Tree(NODETYPE.DELTA, f'delta_{originalDag.companion.new_delta()}', companion=originalDag.companion)   # Derivative of the top node y with respect to itself
    diffDag.rank = originalRank * 2
    diffDag.axes = originalDag.axes + originalDag.axes
    diffDag = _reverse_mode_diff(originalDag, diffDag, arg, originalDag.rank)
    diffDag.set_tensorrank(variable_ranks, arg)
    diffDag.add_incoming_edges()
    diffDag.unify_axes()
    diffDag.rename_equivalent_constants()
    diffDag = diffDag.remove_unneccessary_deltas()
    _simplify(diffDag)
    diffDag.eliminate_common_subtrees()
    diffDag.add_incoming_edges()
    diffDag.set_tensorrank(variable_ranks, arg)
    diffDag.unify_axes()
    diffDag.remove_nonexistant_axes()
    return diffDag, originalDag, arg_name, variable_ranks

def print_axes_help(diffDag):
    print(f'Axis Origins: {diffDag.companion.axis_to_origin}')
    print(f'Variable and Constant Axes:')
    done_nodes = []
    for node in diffDag.get_all_subtrees():
        if node and (not node in done_nodes) and (node.type == NODETYPE.VARIABLE or node.type == NODETYPE.CONSTANT or node.type == NODETYPE.DELTA):
            print(f'{node.name} {node.axes}')
            done_nodes.append(node)
    for constant in diffDag.companion.printing_constants.keys():
        print(f'{constant} {diffDag.companion.printing_constants[constant]}')

def _preprocess(originalDag, arg_name, variable_ranks):
    originalDag.eliminate_common_subtrees()
    arg = originalDag.find(arg_name)
    originalDag = originalDag.fix_missing_indices(arg)
    originalDag.add_incoming_edges()
    originalDag.set_tensorrank(variable_ranks, arg)
    arg = originalDag.find(arg_name)
    originalDag = _split_double_powers(originalDag, arg)
    originalDag = _split_adj(originalDag)
    originalDag.add_incoming_edges()
    originalDag.set_tensorrank(variable_ranks, arg)
    originalDag.unify_axes()
    arg = originalDag.find(arg_name) # Call this again since arg-subtree may have changed
    return originalDag, arg

def _split_double_powers(dag, arg):
    def create_split_power(node):
        indices = ''.join([i for i in string.ascii_lowercase][0:node.left.rank])
        prod = Tree(NODETYPE.PRODUCT, f'*(,{indices}->{indices})', node.right, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'log', None, node.left, companion=node.companion), companion=node.companion)
        prod.set_indices('', indices, indices)
        return Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, prod, companion=node.companion)

    if dag.type == NODETYPE.POWER and dag.left.contains(arg) and dag.right.contains(arg):
        dag = create_split_power(dag)

    def split_powers_helper(node):
        if node.left: split_powers_helper(node.left)
        if node.right: split_powers_helper(node.right)
        if node.left and node.left.type == NODETYPE.POWER and node.left.left.contains(arg) and node.left.right.contains(arg):
            node.left = create_split_power(node.left)
        if node.right and node.right.type == NODETYPE.POWER and node.right.left.contains(arg) and node.right.right.contains(arg):
            node.right = create_split_power(node.right)
    split_powers_helper(dag)
    return dag

def _split_adj(dag):
    if dag.type == NODETYPE.SPECIAL_FUNCTION and dag.name == 'adj':
        dag = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, dag.right, companion=dag.companion), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, dag.right, companion=dag.companion), companion=dag.companion)
        dag.set_indices('', 'ij', 'ij')
    def split_adj_helper(node):
        if node.left: split_adj_helper(node.left)
        if node.right: split_adj_helper(node.right)
        if node.left and node.left.type == NODETYPE.SPECIAL_FUNCTION and node.left.name == 'adj':
            node.left = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, node.left.right, companion=node.companion), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, node.left.right, companion=node.companion), companion=node.companion)
            node.left.set_indices('', 'ij', 'ij')
        if node.right and node.right.type == NODETYPE.SPECIAL_FUNCTION and node.right.name == 'adj':
            node.right = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, node.right.right, companion=node.companion), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, node.right.right, companion=node.companion), companion=node.companion)
            node.right.set_indices('', 'ij', 'ij')
    split_adj_helper(dag)
    return dag

def _reverse_mode_diff(node, diff, arg, yRank):  # Computes derivative of node.left and node.right | node: node in original dag | diff : node that contains derivative with respect to node.
    if node.type == NODETYPE.PRODUCT:
        diff = _diff_product(node, diff, arg, yRank)
    elif node.type == NODETYPE.SUM:
        diff = _diff_sum(node, diff, arg, yRank)
    elif node.type == NODETYPE.POWER:
        diff = _diff_power(node, diff, arg, yRank)
    elif node.type == NODETYPE.ELEMENTWISE_FUNCTION:
        diff = _diff_elementwise_function(node, diff, arg, yRank)
    elif node.type == NODETYPE.SPECIAL_FUNCTION:
        diff = _diff_special_function(node, diff, arg, yRank)
    elif node.type == NODETYPE.VARIABLE:
        diff = _diff_variable(node, diff, arg)
    return diff

def _diff_product(node, diff, arg, yRank):
    currentDiffNode = diff
    s1 = node.leftIndices
    s2 = node.rightIndices
    s3 = node.resultIndices
    s4 = ''.join([i for i in string.ascii_lowercase if i not in (s1 + s2 + s3)][0:yRank])   # Use some unused indices for the output node y
    if node.left and node.left.contains(arg):
        diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s2}->{s4+s1})', currentDiffNode, node.right, companion=currentDiffNode.companion)   # Diff rule
        diff.set_indices(s4+s3, s2, s4+s1)
        diff = _contributions(node.left, diff, arg, yRank)
    if node.right and node.right.contains(arg):
        diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s1}->{s4+s2})', currentDiffNode, node.left, companion=currentDiffNode.companion)
        diff.set_indices(s4+s3, s1, s4+s2)
        diff = _contributions(node.right, diff, arg, yRank)
    return diff

def _diff_sum(node, diff, arg, yRank):
    currentDiffNode = diff
    if node.left and node.left.contains(arg):
        diff = _contributions(node.left, currentDiffNode, arg, yRank)
    if node.right and node.right.contains(arg):
        diff = _contributions(node.right, currentDiffNode, arg, yRank)
    return diff

def _diff_power(node, diff, arg, yRank):
    if node.left and node.right and node.left.contains(arg) and node.right.contains(arg):
        raise Exception('Encountered power node with argument in left and right operands during differentiation.')  # This case is handled by a previous transform of the expression
    if node.left and node.left.contains(arg):
        indices = ''.join([i for i in string.ascii_lowercase][0:node.left.rank])
        one = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        one.rank = 0
        newpower = Tree(NODETYPE.POWER, '^', node.left, Tree(NODETYPE.SUM, '+', node.right, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, one, companion=diff.companion), companion=diff.companion), companion=diff.companion)
        funcDiff = Tree(NODETYPE.PRODUCT, f'*(,{indices}->{indices})', node.right, newpower, companion=diff.companion)
        funcDiff.set_indices('', indices, indices)
        s1 = ''.join(string.ascii_lowercase[0:node.left.rank])   # Same procedure as with an elementwise function
        s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:yRank])
        diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff, companion=diff.companion) # Diff rule
        diff.set_indices(s2+s1, s1, s2+s1)
        diff = _contributions(node.left, diff, arg, yRank)
    elif node.right and node.right.contains(arg):
        s3 = ''.join([i for i in string.ascii_lowercase][0:yRank])
        s2 = ''.join([i for i in string.ascii_lowercase if not i in s3][0:node.rank])
        s1 = ''
        funcDiff = Tree(NODETYPE.PRODUCT, f'*({s2},{s2}->{s2})', node, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'log', None, node.left, companion=diff.companion), companion=diff.companion)
        funcDiff.set_indices(s2, s2, s2)
        diff = Tree(NODETYPE.PRODUCT, f'*({s3+s2},{s2+s1}->{s3+s1})', diff, funcDiff, companion=diff.companion)
        diff.set_indices(s3+s2, s2+s1, s3+s1)
        diff = _contributions(node.right, diff, arg, yRank)
    return diff

def _diff_elementwise_function(node, diff, arg, yRank):
    if node.name == '-':
        const = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        const.rank = node.right.rank
        const.axes = node.right.axes
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, const, companion=diff.companion)
    elif node.name == 'sin':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right, companion=diff.companion)
    elif node.name == 'cos':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sin', None, node.right, companion=diff.companion), companion=diff.companion)
    elif node.name == 'tan':
        cos = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right, companion=diff.companion)
        indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
        cos_squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', cos, cos, companion=diff.companion)
        cos_squared.set_indices(indices, indices, indices)
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, cos_squared, companion=diff.companion)
    elif node.name == 'arcsin':
        const1 = Tree(NODETYPE.CONSTANT, f'2_{diff.companion.new_constant()}', companion=diff.companion)
        const1.rank = node.right.rank
        const1.axes = node.right.axes
        x_squared = Tree(NODETYPE.POWER, '^', node.right, const1)
        const2 = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        const2.rank = node.right.rank
        const2.axes = node.right.axes
        inside_root = Tree(NODETYPE.SUM, '+', const2, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, x_squared, companion=diff.companion), companion=diff.companion)
        const3 = Tree(NODETYPE.CONSTANT, f'0.5_{diff.companion.new_constant()}', companion=diff.companion)
        const3.rank = 0
        const3.axes = []
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.POWER, '^', inside_root, const3, companion=diff.companion), companion=diff.companion)
    elif node.name == 'arccos':
        const1 = Tree(NODETYPE.CONSTANT, f'2_{diff.companion.new_constant()}', companion=diff.companion)
        const1.rank = node.right.rank
        const1.axes = node.right.axes
        x_squared = Tree(NODETYPE.POWER, '^', node.right, const1, companion=diff.companion)
        const2 = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        const2.rank = node.right.rank
        const2.axes = node.right.axes
        inside_root = Tree(NODETYPE.SUM, '+', const2, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, x_squared, companion=diff.companion), companion=diff.companion)
        const3 = Tree(NODETYPE.CONSTANT, f'0.5_{diff.companion.new_constant()}', companion=diff.companion)
        const3.rank = 0
        const3.axes = []
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.POWER, '^', inside_root, const3, companion=diff.companion), companion=diff.companion), companion=diff.companion)
    elif node.name == 'arctan':
        indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
        squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node.right, node.right, companion=diff.companion)
        squared.set_indices(indices, indices, indices)
        const = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        const.rank = node.right.rank
        const.axes = node.right.axes
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.SUM, '+', squared, const, companion=diff.companion), companion=diff.companion)
    elif node.name == 'exp':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, node.right, companion=diff.companion)
    elif node.name == 'log':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, node.right, companion=diff.companion)
    elif node.name == 'tanh':
        indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
        squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node, node, companion=diff.companion)
        squared.set_indices(indices, indices, indices)
        const = Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion)
        const.rank = node.right.rank
        const.axes = node.right.axes
        funcDiff = Tree(NODETYPE.SUM, '+', const, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, squared, companion=diff.companion), companion=diff.companion)
    elif node.name == 'abs':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sign', None, node.right, companion=diff.companion)
    elif node.name == 'sign':
        funcDiff = Tree(NODETYPE.CONSTANT, f'0_{diff.companion.new_constant()}', companion=diff.companion)
        funcDiff.rank = node.right.rank
        funcDiff.axes = node.right.axes
    elif node.name == 'relu':
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'relu', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sign', None, node.right, companion=diff.companion), companion=diff.companion)
    elif node.name == 'elementwise_inverse':
        indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
        squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node.right, node.right, companion=diff.companion)
        squared.set_indices(indices, indices, indices)
        funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, squared, companion=diff.companion), companion=diff.companion)
    else:
        raise Exception(f'Unknown function {node.name} encountered during differentiation.')
    s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
    s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:yRank])
    diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff, companion=diff.companion) # Diff rule
    diff.set_indices(s2+s1, s1, s2+s1)
    if node.right and node.right.contains(arg):
        diff = _contributions(node.right, diff, arg, yRank)
    return diff

def _diff_special_function(node, diff, arg, yRank):
    if node.name == 'inv':
        funcDiff = Tree(NODETYPE.PRODUCT, f'*(ij,kl->kjli)', Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, node, companion=diff.companion), node, companion=diff.companion)
        funcDiff.set_indices('ij', 'kl', 'kjli')
    if node.name == 'det':
        funcDiff = Tree(NODETYPE.PRODUCT, '*(ij,->ji)', Tree(NODETYPE.SPECIAL_FUNCTION, 'adj', None, node.right, companion=diff.companion), Tree(NODETYPE.CONSTANT, f'1_{diff.companion.new_constant()}', companion=diff.companion), companion=diff.companion)
        funcDiff.set_indices('ij', '', 'ji')
    s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
    s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:node.rank])
    s3 = ''.join([i for i in string.ascii_lowercase if i not in s1+s2][0:yRank])
    diff = Tree(NODETYPE.PRODUCT, f'*({s3+s2},{s2+s1}->{s3+s1})', diff, funcDiff, companion=diff.companion) # Diff rule
    diff.set_indices(s3+s2, s2+s1, s3+s1)
    if node.right and node.right.contains(arg):
        diff = _contributions(node.right, diff, arg, yRank)
    return diff

def _diff_variable(node, diff, arg):
    if node == arg:
        pass
    else:
        raise Exception('Reached non-argument variable during differentiation.')
    return diff
    
def _contributions(node, diff, arg, yRank):
    global _originalNodeToDiffNode
    global _originalNodeToDiffTree
    if node in _originalNodeToDiffNode:   # If we've been to this node before, we need to add the new contribution to the old one
        diff = Tree(NODETYPE.SUM, '+', _originalNodeToDiffNode[node], diff, companion=diff.companion)
        savedDiffNode = _originalNodeToDiffNode[node]   # Need this in a second
        _originalNodeToDiffNode[node] = diff   # Future contributions need to be added here
        # When we add a new contribution to dY/dX (diff), we also need to incorporate this contribution in dX/dZ for any child nodes Z of X
        _originalNodeToDiffTree[node].add_incoming_edges()
        for n in savedDiffNode.incoming:
            if n.left == savedDiffNode:
                n.left = diff
            elif n.right == savedDiffNode:
                n.right = diff
        if len(savedDiffNode.incoming) != 0:
            diff = _originalNodeToDiffTree[node]
    else:
        _originalNodeToDiffNode[node] = diff
        diff = _reverse_mode_diff(node, diff, arg, yRank)
        _originalNodeToDiffTree[node] = diff
    return diff

def _simplify(node):
    if node.left: _simplify(node.left)
    if node.right: _simplify(node.right)
    node_changed = True
    while(node_changed):
        node_changed = False
        if _is_simplifiable_sum_minus_1(node): # Simplify (a + (- b)) to (a - b)
            node.type = NODETYPE.DIFFERENCE
            node.name = '-'
            node.right = node.right.right
            node_changed = True
        if _is_simplifiable_power(node): # Simplify exp(b * log(a)) to (a ^ b)
            node.type = NODETYPE.POWER
            node.name = '^'
            node.left = node.right.left
            node.right = node.right.right.right
            node_changed = True
        if _is_simplifiable_adj(node): # Simplify det(X) * inv(X) = adj(X)
            node.type = NODETYPE.SPECIAL_FUNCTION
            node.name = 'adj'
            node.left = None
            node.right = node.right.right
            node_changed = True
        if _is_simplifiable_const_minus(node): # Turn (- (const)) into a const with a minus
            node.type = NODETYPE.CONSTANT
            if node.right.name.startswith('-'):
                node.name = node.right.name.strip('-')
            else:
                node.name = '-' + node.right.name
            node.left = None
            node.right = None
            node_changed = True
        if _is_simplifiable_const_sum(node): # Compute sum of constants
            node.type = NODETYPE.CONSTANT
            node.name = str(int(node.left.name.split('_')[0]) + int(node.right.name.split('_')[0]))
            node.left = None
            node.right = None
            node_changed = True
        if _is_simplifiable_const_sum_minus(node): # Simplify (a + b) to (a - (-b)) when b is a const and has a -
            node.type = NODETYPE.DIFFERENCE
            node.name = '-'
            node.right.name = node.right.name.strip('-')
            node_changed = True
        if _is_simplifiable_const_diff(node): # Compute difference of constants
            node.type = NODETYPE.CONSTANT
            node.name = str(int(node.left.name.split('_')[0]) - int(node.right.name.split('_')[0]))
            node.left = None
            node.right = None
            node_changed = True

def _is_simplifiable_sum_minus_1(node):
    return  node.type == NODETYPE.SUM and \
            node.right.type == NODETYPE.ELEMENTWISE_FUNCTION and \
            node.right.name == '-'

def _is_simplifiable_power(node):
    return  node.type == NODETYPE.ELEMENTWISE_FUNCTION and \
            node.name == 'exp' and \
            node.right.type == NODETYPE.PRODUCT and \
            node.right.left.rank == 0 and \
            node.right.rightIndices == node.right.resultIndices and \
            node.right.right.type == NODETYPE.ELEMENTWISE_FUNCTION and \
            node.right.right.name == 'log'

def _is_simplifiable_adj(node):
    return  node.type == NODETYPE.PRODUCT and \
            node.left.type == NODETYPE.SPECIAL_FUNCTION and \
            node.right.type == NODETYPE.SPECIAL_FUNCTION and \
            node.left.name == 'det' and \
            node.right.name == 'inv' and \
            node.left.right == node.right.right

def _is_simplifiable_const_minus(node):
    return  node.type == NODETYPE.ELEMENTWISE_FUNCTION and \
            node.name == '-' and \
            node.right.type == NODETYPE.CONSTANT

def _is_simplifiable_const_sum(node):
    return  node.type == NODETYPE.SUM and \
            node.left.type == NODETYPE.CONSTANT and \
            node.right.type == NODETYPE.CONSTANT

def _is_simplifiable_const_sum_minus(node):
    return  node.type == NODETYPE.SUM and \
            node.right.type == NODETYPE.CONSTANT and \
            node.right.name.startswith('-')
        
def _is_simplifiable_const_diff(node):
    return  node.type == NODETYPE.DIFFERENCE and \
            node.left.type == NODETYPE.CONSTANT and \
            node.right.type == NODETYPE.CONSTANT

if __name__ == '__main__':
    example= '''
        declare x 1 expression tanh(x) derivative wrt x
    '''
    diffDag, originalDag, arg_name, variable_ranks = differentiate(example)
    print(diffDag)
    print_axes_help(diffDag)

