from parser import *

class Differentiator():
    def __init__(self, input):
        self.input = input
        self.parser = Parser(input)
        self.parser.parse()
        self.parser.eliminate_common_subtrees()
        self.dag = self.parser.tree
        self.variable_ranks = self.parser.variable_ranks
        self.dag.set_tensorrank(self.variable_ranks)
        self.arg = self.dag.find(self.parser.arg_name)
        self.diffDag = None

    def arg_check(self):
        if self.parser.arg_name not in self.variable_ranks:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in declared variables.')
        if not self.arg:
            raise Exception('Argument \'{self.parser.arg_name}\' not found in expression.')

    def differentiate(self):
        y = self.dag
        self.diffDag = Tree(NODETYPE.VARIABLE, 'I') # Derivative of the top node y with respect to itself
        self.variable_ranks['I'] = y.rank
        def reverse_mode_diff(node):
            if node.type == NODETYPE.PRODUCT:
                s1 = node.leftIndices
                s2 = node.rightIndices
                s3 = node.resultIndices
                s4 = ''.join([i for i in string.ascii_lowercase if i not in (s1 + s2 + s3)][0:y.rank])  # Use some unused indices for the output node y
                if node.left and node.left.contains(self.arg):
                    self.diffDag = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s2}->{s4+s1})', self.diffDag, node.right)
                    self.diffDag.set_indices(s4+s3, s2, s4+s1)
                    reverse_mode_diff(node.left)
                if node.right and node.right.contains(self.arg):
                    self.diffDag = Tree(NODETYPE.PRODUCT, f'*({s4+s3},{s1}->{s4+s2})', self.diffDag, node.left)
                    self.diffDag.set_indices(s4+s3, s1, s4+s2)
                    reverse_mode_diff(node.right)
            elif node.type == NODETYPE.VARIABLE:
                if node == self.arg:
                    pass
                else:
                    print("HUUUHH??")
        reverse_mode_diff(self.dag)
    
    def print(self):
        self.diffDag.dot('diffdag')

if __name__ == '__main__':
    example = 'declare a 1 b 1 c 0 argument a expression (a*(i,i->)b)*(,->)c'
    d = Differentiator(example)
    d.differentiate()
    d.diffDag.set_tensorrank(d.variable_ranks)
    d.print()