from parser import Parser
from scanner import ELEMENTWISE_FUNCTIONS, TOKEN_ID
from tree import Tree, NODETYPE
import string

# This class differentiates an expression dag with respect to an argument supplied by the parser
class Differentiator():
    def __init__(self, input):
        self.input = input
        parser = Parser(input)
        parser.parse()
        self.originalDag = parser.dag
        self.variable_ranks = parser.variable_ranks
        self.arg = parser.arg
        self.diffDag = None
        self.originalNodeToDiffNode = {}   # For a node X, we save where dY/dX is, to allow adding more chain rule contributions when we reach that node again later
        self.originalNodeToDiffTree = {}   # For a node X, we also the tree which contains dX/dZ (for possibly 2 nodes Z), which also we need to update when adding chain rule contributions

    def arg_check(self):
        if self.parser.arg_name not in self.variable_ranks:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in declared variables.')
        if not self.arg:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in expression.')

    def differentiate(self):
        deltaRank = self.originalDag.rank * 2
        self.diffDag = Tree(NODETYPE.VARIABLE, f'_delta({deltaRank})')   # Derivative of the top node y with respect to itself, unnecessary ones will later be removed
        self.variable_ranks[f'_delta({deltaRank})'] = deltaRank
        self.diffDag = self.reverse_mode_diff(self.originalDag, self.diffDag)
        self.diffDag.set_tensorrank(self.variable_ranks, self.arg)

    def reverse_mode_diff(self, node, diff):  # Computes derivative of node.left and node.right | node: node in original dag | diff : node that contains derivative with respect to node.
        if node.type == NODETYPE.PRODUCT:
            diff = self.diff_product(node, diff)
        elif node.type == NODETYPE.SUM:
            diff = self.diff_sum(node, diff)
        elif node.type == NODETYPE.POWER:
            diff = self.diff_power(node, diff)
        elif node.type == NODETYPE.ELEMENTWISE_FUNCTION:
            diff = self.diff_elementwise_function(node, diff)
        elif node.type == NODETYPE.SPECIAL_FUNCTION:
            diff = self.diff_special_function(node, diff)
        elif node.type == NODETYPE.VARIABLE:
            diff = self.diff_variable(node, diff)
        return diff
    
    def diff_product(self, node, diff):
        currentDiffNode = diff
        s1 = node.leftIndices
        s2 = node.rightIndices
        s3 = node.resultIndices
        s4 = ''.join([i for i in string.ascii_lowercase if i not in (s1 + s2 + s3)][0:self.originalDag.rank])   # Use some unused indices for the output node y
        if node.left and node.left.contains(self.arg):
            diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s2}->{s4+s1})', currentDiffNode, node.right)   # Diff rule
            diff.set_indices(s4+s3, s2, s4+s1)
            diff = self.contributions(node.left, diff)
        if node.right and node.right.contains(self.arg):
            diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s1}->{s4+s2})', currentDiffNode, node.left)
            diff.set_indices(s4+s3, s1, s4+s2)
            diff = self.contributions(node.right, diff)
        return diff
    
    def diff_sum(self, node, diff):
        currentDiffNode = diff
        if node.left and node.left.contains(self.arg):
            diff = self.contributions(node.left, currentDiffNode)
        if node.right and node.right.contains(self.arg):
            diff = self.contributions(node.right, currentDiffNode)
        return diff
    
    def diff_power(self, node, diff):
        if node.left and node.right and node.left.contains(self.arg) and node.right.contains(self.arg):
            raise Exception('Encountered power node with argument in left and right operands during differentiation.')  # This case is handled by a previous transform of the expression
        if node.left and node.left.contains(self.arg):
            indices = ''.join([i for i in string.ascii_lowercase][0:node.left.rank])
            one = Tree(NODETYPE.CONSTANT, '1')
            one.rank = 0
            newpower = Tree(NODETYPE.POWER, '^', node.left, Tree(NODETYPE.SUM, '+', node.right, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, one)))
            funcDiff = Tree(NODETYPE.PRODUCT, f'*(,{indices}->{indices})', node.right, newpower)
            funcDiff.set_indices('', indices, indices)
            s1 = ''.join(string.ascii_lowercase[0:node.left.rank])   # Same procedure as with an elementwise function
            s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:self.originalDag.rank])
            diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff) # Diff rule
            diff.set_indices(s2+s1, s1, s2+s1)
            diff = self.contributions(node.left, diff)
        elif node.right and node.right.contains(self.arg):
            s3 = ''.join([i for i in string.ascii_lowercase][0:self.originalDag.rank])
            s2 = ''.join([i for i in string.ascii_lowercase if not i in s3][0:node.rank])
            s1 = ''
            funcDiff = Tree(NODETYPE.PRODUCT, f'*({s2},{s2}->{s2})', node, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'log', None, node.left))
            funcDiff.set_indices(s2, s2, s2)
            diff = Tree(NODETYPE.PRODUCT, f'*({s3+s2},{s2+s1}->{s3+s1})', diff, funcDiff)
            diff.set_indices(s3+s2, s2+s1, s3+s1)
            diff = self.contributions(node.right, diff)
        return diff
    
    def diff_elementwise_function(self, node, diff):
        if node.name == '-':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.CONSTANT, f'1'))
        elif node.name == 'sin':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right)
        elif node.name == 'cos':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sin', None, node.right))
        elif node.name == 'tan':
            cos = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right)
            indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
            cos_squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', cos, cos)
            cos_squared.set_indices(indices, indices, indices)
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, cos_squared)
        elif node.name == 'arcsin':
            x_squared = Tree(NODETYPE.POWER, '^', node.right, Tree(NODETYPE.CONSTANT, 2))
            inside_root = Tree(NODETYPE.SUM, '+', Tree(NODETYPE.CONSTANT, f'1'), Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, x_squared))
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.POWER, '^', inside_root, Tree(NODETYPE.CONSTANT, 0.5)))
        elif node.name == 'arccos':
            x_squared = Tree(NODETYPE.POWER, '^', node.right, Tree(NODETYPE.CONSTANT, 2))
            inside_root = Tree(NODETYPE.SUM, '+', Tree(NODETYPE.CONSTANT, f'1'), Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, x_squared))
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.POWER, '^', inside_root, Tree(NODETYPE.CONSTANT, 0.5))))
        elif node.name == 'arctan':
            indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
            squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node.right, node.right)
            squared.set_indices(indices, indices, indices)
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.SUM, '+', squared, Tree(NODETYPE.CONSTANT, f'1')))
        elif node.name == 'exp':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, node.right)
        elif node.name == 'log':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, node.right)
        elif node.name == 'tanh':
            indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
            squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node, node)
            squared.set_indices(indices, indices, indices)
            funcDiff = Tree(NODETYPE.SUM, '+', Tree(NODETYPE.CONSTANT, f'1'), Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, squared))
        elif node.name == 'abs':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sign', None, node.right)
        elif node.name == 'sign':
            funcDiff = Tree(NODETYPE.CONSTANT, f'0')
            funcDiff.rank = node.right.rank
        elif node.name == 'relu':
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'relu', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sign', None, node.right))
        elif node.name == 'elementwise_inverse':
            indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
            squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node.right, node.right)
            squared.set_indices(indices, indices, indices)
            funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, squared))
        else:
            raise Exception(f'Unknown function {node.name} encountered during differentiation.')
        s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
        s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:self.originalDag.rank])
        diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff) # Diff rule
        diff.set_indices(s2+s1, s1, s2+s1)
        if node.right and node.right.contains(self.arg):
            diff = self.contributions(node.right, diff)
        return diff
    
    def diff_special_function(self, node, diff):
        if node.name == 'inv':
            funcDiff = Tree(NODETYPE.PRODUCT, f'*(ij,kl->ijkl)', Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, node), node)
            funcDiff.set_indices('ij', 'kl', 'ijkl')
        if node.name == 'det':
            funcDiff = Tree(NODETYPE.PRODUCT, '*(ij,->ji)', Tree(NODETYPE.SPECIAL_FUNCTION, 'adj', None, node.right), Tree(NODETYPE.CONSTANT, '1'))
            funcDiff.set_indices('ij', '', 'ji')
        s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
        s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:node.rank])
        s3 = ''.join([i for i in string.ascii_lowercase if i not in s1+s2][0:self.originalDag.rank])
        diff = Tree(NODETYPE.PRODUCT, f'*({s3+s2},{s2+s1}->{s3+s1})', diff, funcDiff) # Diff rule
        diff.set_indices(s3+s2, s2+s1, s3+s1)
        if node.right and node.right.contains(self.arg):
            diff = self.contributions(node.right, diff)
        return diff
    
    def diff_variable(self, node, diff):
        if node == self.arg:
            pass
        else:
            raise Exception('Reached non-argument variable during differentiation.')
        return diff
        
    def contributions(self, node, diff):
        if node in self.originalNodeToDiffNode:   # If we've been to this node before, we need to add the new contribution to the old one
            diff = Tree(NODETYPE.SUM, '+', self.originalNodeToDiffNode[node], diff)
            savedDiffNode = self.originalNodeToDiffNode[node]   # Need this in a second
            self.originalNodeToDiffNode[node] = diff   # Future contributions need to be added here
            # When we add a new contribution to dY/dX (diff), we also need to incorporate this contribution in dX/dZ for any child nodes Z of X
            self.originalNodeToDiffTree[node].add_incoming_edges()
            for n in savedDiffNode.incoming:
                if n.left == savedDiffNode:
                    n.left = diff
                elif n.right == savedDiffNode:
                    n.right = diff
            if len(savedDiffNode.incoming) != 0:
                diff = self.originalNodeToDiffTree[node]
        else:
            self.originalNodeToDiffNode[node] = diff
            diff = self.reverse_mode_diff(node, diff)
            self.originalNodeToDiffTree[node] = diff
        return diff

    def render(self, filename='diffdag'):
        self.diffDag.dot(filename)

if __name__ == '__main__':
    example = '''
    declare 
        X 2
    expression det(X)
    derivative wrt X
    '''
    d = Differentiator(example)
    d.differentiate()
    d.render()
    print(d.diffDag)