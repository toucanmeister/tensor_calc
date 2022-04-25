from parser import Parser
from scanner import TOKEN_ID
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

    def arg_check(self):
        if self.parser.arg_name not in self.variable_ranks:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in declared variables.')
        if not self.arg:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in expression.')

    def differentiate(self):
        y = self.originalDag
        self.diffDag = Tree(NODETYPE.VARIABLE, '_IDENTITY')   # Derivative of the top node y with respect to itself, unnecessary ones will later be removed
        self.variable_ranks['_IDENTITY'] = y.rank
        originalNodeToDiffNode = {}   # Saves where in the dag the diff of a node is, to allow adding more chain rule contributions when we reach that node again later
        
        def reverse_mode_diff(node, diff):  # Computes derivative of node.left and node.right | node: node in original dag | diff : node that contains derivative with respect to node.
            # PRODUCT
            currentDiffNode = diff
            if node.type == NODETYPE.PRODUCT:
                s1 = node.leftIndices
                s2 = node.rightIndices
                s3 = node.resultIndices
                s4 = ''.join([i for i in string.ascii_lowercase if i not in (s1 + s2 + s3)][0:y.rank])   # Use some unused indices for the output node y
                if node.left and node.left.contains(self.arg):
                    diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s2}->{s4+s1})', currentDiffNode, node.right)   # Diff rule
                    diff.set_indices(s4+s3, s2, s4+s1)
                    if node.left in originalNodeToDiffNode:   # If we've been to this node before, we need to add the new contribution to the old one
                        diff = Tree(NODETYPE.SUM, '+', originalNodeToDiffNode[node.left], diff)
                        originalNodeToDiffNode[node.left] = diff
                    else:
                        originalNodeToDiffNode[node.left] = diff
                        diff = reverse_mode_diff(node.left, diff)
                if node.right and node.right.contains(self.arg):
                    diff = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s1}->{s4+s2})', currentDiffNode, node.left)
                    diff.set_indices(s4+s3, s1, s4+s2)
                    if node.right in originalNodeToDiffNode:
                        diff = Tree(NODETYPE.SUM, '+', originalNodeToDiffNode[node.right], diff)
                        originalNodeToDiffNode[node.right] = diff
                    else:
                        originalNodeToDiffNode[node.right] = diff
                        diff = reverse_mode_diff(node.right, diff)
            # SUM
            elif node.type == NODETYPE.SUM:
                if node.left and node.left.contains(self.arg):
                    if node.left in originalNodeToDiffNode:
                        diff = Tree(NODETYPE.SUM, '+', originalNodeToDiffNode[node.left], diff)
                        originalNodeToDiffNode[node.left] = diff
                    else: 
                        originalNodeToDiffNode[node.left] = diff
                        diff = reverse_mode_diff(node.left, currentDiffNode)
                if node.right and node.right.contains(self.arg):
                    if node.right in originalNodeToDiffNode:
                        diff = Tree(NODETYPE.SUM, '+', originalNodeToDiffNode[node.right], diff)
                        originalNodeToDiffNode[node.right] = diff
                    else:
                        originalNodeToDiffNode[node.right] = diff
                        diff = reverse_mode_diff(node.right, currentDiffNode)
            # ELEMENTWISE FUNCTION
            elif node.type == NODETYPE.ELEMENTWISE_FUNCTION:
                if node.name == '-':
                    funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.VARIABLE, '_IDENTITY'))
                elif node.name == 'sin':
                    funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right)
                elif node.name == 'cos':
                    funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sin', None, node.right))
                else:
                    raise Exception(f'Unknown function {node.name} encountered during differentiation.')
                s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
                s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:y.rank])
                diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff) # Diff rule
                if node.right in originalNodeToDiffNode:
                    diff = Tree(NODETYPE.SUM, '+', originalNodeToDiffNode[node.right], diff)
                    originalNodeToDiffNode[node.right] = diff
                else:
                    originalNodeToDiffNode[node.right] = diff
                    diff = reverse_mode_diff(node.right, diff)
            elif node.type == NODETYPE.VARIABLE:
                if node == self.arg:
                    pass
                else:
                    raise Exception('Reached non-argument variable during differentiation.')
            return diff

        self.diffDag = reverse_mode_diff(self.originalDag, self.diffDag)
        self.remove_identity()

    def render(self):
        self.diffDag.dot('diffdag')

    def remove_identity(self):   # Removes the unnecessary _IDENTITY nodes we added for convenvience during differentiation
        if self.diffDag.type == NODETYPE.PRODUCT: # In the top node
            if self.diffDag.left.name == '_IDENTITY':
                self.diffDag = self.diffDag.right
            elif self.diffDag.right.name == '_IDENTITY':
                self.diffDag = self.diffDag.left
        def helper(node): # In the rest of the DAG
            if not node:
                return
            if node.left and node.left.type == NODETYPE.PRODUCT:
                if node.left.left and node.left.left.name == '_IDENTITY':
                    node.left = node.left.right
                elif node.left.right and node.left.right.name == '_IDENTITY':
                    node.left = node.left.left
            if node.right and node.right.type == NODETYPE.PRODUCT:
                if node.right.left and node.right.left.name == '_IDENTITY':
                    node.right = node.right.right
                elif node.right.right and node.right.right.name == '_IDENTITY':
                    node.right = node.right.left
            helper(node.left)
            helper(node.right)
        helper(self.diffDag)

if __name__ == '__main__':
    example = 'declare x 1 argument x expression sin(x)'
    d = Differentiator(example)
    d.differentiate()
    print(d.diffDag)