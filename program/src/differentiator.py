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
        self.originalNodeToDiffNode = {}   # For a node X, we save where dY/dX is, to allow adding more chain rule contributions when we reach that node again later
        self.originalNodeToDiffTree = {}   # For a node X, we also the tree which contains dX/dZ (for possibly 2 nodes Z), which also we need to update when adding chain rule contributions
        self.identityCounter = 1   # Need a way to differentiate different Identity (1) nodes added, this keeps track to give them different IDs

    def arg_check(self):
        if self.parser.arg_name not in self.variable_ranks:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in declared variables.')
        if not self.arg:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in expression.')

    def differentiate(self):
        self.diffDag = Tree(NODETYPE.VARIABLE, '_IDENTITY_0')   # Derivative of the top node y with respect to itself, unnecessary ones will later be removed
        self.variable_ranks['_IDENTITY_0'] = self.originalDag.rank
        self.diffDag = self.reverse_mode_diff(self.originalDag, self.diffDag)
        self.remove_identity()

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
                idName = '_IDENTITY_' + str(self.identityCounter)
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.VARIABLE, idName))
                self.identityCounter = self.identityCounter + 1
                self.variable_ranks[idName] = node.right.rank
            elif node.name == 'sin':
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'cos', None, node.right)
            elif node.name == 'cos':
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'sin', None, node.right))
            elif node.name == 'exp':
                funcDiff = Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, node.right)
            else:
                raise Exception(f'Unknown function {node.name} encountered during differentiation.')
            s1 = ''.join(string.ascii_lowercase[0:node.right.rank])
            s2 = ''.join([i for i in string.ascii_lowercase if i not in s1][0:self.originalDag.rank])
            diff = Tree(NODETYPE.PRODUCT, f'*({s2+s1},{s1}->{s2+s1})', diff, funcDiff) # Diff rule
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

    def remove_identity(self):   # Removes the unnecessary _IDENTITY nodes we added for convenvience during differentiation
        if self.diffDag.type == NODETYPE.PRODUCT: # In the top node
            if self.diffDag.left.name.startswith('_IDENTITY'):
                self.diffDag = self.diffDag.right
            elif self.diffDag.right.name.startswith('_IDENTITY'):
                self.diffDag = self.diffDag.left
        def helper(node): # In the rest of the DAG
            if not node:
                return
            if node.left and node.left.type == NODETYPE.PRODUCT:
                if node.left.left and node.left.left.name.startswith('_IDENTITY'):
                    node.left = node.left.right
                elif node.left.right and node.left.right.name.startswith('_IDENTITY'):
                    node.left = node.left.left
            if node.right and node.right.type == NODETYPE.PRODUCT:
                if node.right.left and node.right.left.name.startswith('_IDENTITY'):
                    node.right = node.right.right
                elif node.right.right and node.right.right.name.startswith('_IDENTITY'):
                    node.right = node.right.left
            helper(node.left)
            helper(node.right)
        helper(self.diffDag)
        def remove_identity_numbers(node):
            if node.name.startswith('_IDENTITY'):
                node.name = '_IDENTITY'
            if node.left: 
                remove_identity_numbers(node.left)
            if node.right:
                remove_identity_numbers(node.right)
        remove_identity_numbers(self.diffDag)

if __name__ == '__main__':
    example = 'declare x 1 argument x expression x *(i,i->ii) x'
    d = Differentiator(example)
    d.differentiate()
    print(d.diffDag)