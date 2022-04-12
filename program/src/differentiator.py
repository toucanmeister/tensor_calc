from parser import *

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
        self.diffDag = Tree(NODETYPE.VARIABLE, '_IDENTITY') # Derivative of the top node y with respect to itself
        self.variable_ranks['_IDENTITY'] = y.rank
        originalNodeToDiffNode = {} # We save diffNodes here, to sum later in the chain rule
        def reverse_mode_diff(node):
            if node.type == NODETYPE.PRODUCT:
                s1 = node.leftIndices
                s2 = node.rightIndices
                s3 = node.resultIndices
                s4 = ''.join([i for i in string.ascii_lowercase if i not in (s1 + s2 + s3)][0:y.rank])  # Use some unused indices for the output node y
                currentDiffNode = self.diffDag
                if node.left and node.left.contains(self.arg):
                    self.diffDag = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s2}->{s4+s1})', self.diffDag, node.right)
                    self.diffDag.set_indices(s4+s3, s2, s4+s1)
                    if node.left in originalNodeToDiffNode:
                        self.diffDag = Tree(NODETYPE.SUM, '+', self.diffDag, originalNodeToDiffNode[node.left])
                    else:
                        originalNodeToDiffNode[node.left] = self.diffDag
                        reverse_mode_diff(node.left)
                if node.right and node.right.contains(self.arg):
                    self.diffDag = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s1}->{s4+s2})', currentDiffNode, node.left)
                    self.diffDag.set_indices(s4+s3, s1, s4+s2)
                    if node.right in originalNodeToDiffNode:
                        self.diffDag = Tree(NODETYPE.SUM, '+', self.diffDag, originalNodeToDiffNode[node.right])
                    else:
                        originalNodeToDiffNode[node.right] = self.diffDag
                        reverse_mode_diff(node.right)
            elif node.type == NODETYPE.VARIABLE:
                if node == self.arg:
                    pass
                else:
                    raise Exception('I don\'t know how we got here...')
        reverse_mode_diff(self.originalDag)
        self.remove_identity()
    
    def render(self):
        self.diffDag.dot('diffdag')

    def remove_identity(self):
        if self.diffDag.type == NODETYPE.PRODUCT:
            if self.diffDag.left.name == '_IDENTITY':
                self.diffDag = self.diffDag.right
            if self.diffDag.right.name == '_IDENTITY':
                self.diffDag = self.diffDag.left
        def helper(node):
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
                if node.right.right and node.right.right.name == '_IDENTITY':
                    node.right = node.right.left
            helper(node.left)
            helper(node.right)
        helper(self.diffDag)

if __name__ == '__main__':
    example = 'declare a 1 b 1 c 0 argument c expression (a*(i,i->)b) *(,->)c'
    d = Differentiator(example)
    d.differentiate()
    d.diffDag.set_tensorrank(d.variable_ranks)
    d.diffDag.eliminate_common_subtrees()
    d.render()