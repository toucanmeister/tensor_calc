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
        self.arg = self.originalDag.find(parser.arg_name)
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
        self.diffDag.set_tensorrank(self.variable_ranks)

    def reverse_mode_diff(self, node, diff):  # Computes derivative of node.left and node.right | node: node in original dag | diff : node that contains derivative with respect to node.
        currentDiffNode = diff
        # PRODUCT
        if node.type == NODETYPE.PRODUCT:
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
        # SUM
        elif node.type == NODETYPE.SUM:
            if node.left and node.left.contains(self.arg):
                diff = self.contributions(node.left, currentDiffNode)
            if node.right and node.right.contains(self.arg):
                diff = self.contributions(node.right, currentDiffNode)
        # ELEMENTWISE FUNCTION
        elif node.type == NODETYPE.ELEMENTWISE_FUNCTION:
            if node.name == '-':
                ones_rank = node.right.rank
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.VARIABLE, f'_ones({ones_rank})'))
                self.variable_ranks[f'_ones({ones_rank})'] = ones_rank
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
            elif node.name == 'arctan':
                indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
                squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node.right, node.right)
                squared.set_indices(indices, indices, indices)
                ones_rank = node.right.rank
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, Tree(NODETYPE.SUM, '+', squared, Tree(NODETYPE.VARIABLE, f'_ones({ones_rank})')))
                self.variable_ranks[f'_ones({ones_rank})'] = ones_rank
            elif node.name == 'exp':
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, node.right)
            elif node.name == 'log':
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, node.right)
            elif node.name == 'tanh':
                indices = ''.join([i for i in string.ascii_lowercase][0:node.right.rank])
                squared = Tree(NODETYPE.PRODUCT, f'*({indices},{indices}->{indices})', node, node)
                squared.set_indices(indices, indices, indices)
                ones_rank = node.right.rank
                funcDiff = Tree(NODETYPE.SUM, '+', Tree(NODETYPE.VARIABLE, f'_ones({ones_rank})'), Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, squared))
                self.variable_ranks[f'_ones({ones_rank})'] = ones_rank
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
        # VARIABLE
        elif node.type == NODETYPE.VARIABLE:
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
    example = 'declare x 1 argument x expression arctan(x)'
    d = Differentiator(example)
    d.differentiate()
    d.render()
    print(d.diffDag)