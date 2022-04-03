from enum import Enum
from graphviz import Digraph

class NODETYPE(Enum):
    CONSTANT = 'constant'
    VARIABLE = 'variable'
    FUNCTION = 'function'
    ELEMENTWISE_FUNCTION = 'elementwise_function'
    SUM = 'sum'
    DIFFERENCE = 'difference'
    QUOTIENT = 'quotient'
    PRODUCT = 'product'


class Tree():
    running_id = 0
    def __init__(self, nodetype, name, left=None, right=None):
        self.type = nodetype
        self.name = name
        self.left = left
        self.right = right
        self.incoming = []  # Nodes from incoming edges, only filled after common subexpression elimination
        self.rank = -1      # Initialization value
        if self.type == NODETYPE.PRODUCT:    
            self.leftIndices = ''       # Indices in the einstein product notation
            self.rightIndices = ''
            self.resultIndices = ''
        self.id = Tree.running_id   # This is just for the visualization
        Tree.running_id += 1
    
    def set_left(self, left):
        self.left = left
    
    def set_right(self, right):
        self.right = right
    
    def set_name(self, name):
        self.name = name
    
    def set_indices(self, left, right, result):
        self.leftIndices = left
        self.rightIndices = right
        self.resultIndices = result

    def __repr__(self):
        if self.left:
            if self.right:
                return f'{self.left} {self.name} {self.right}'
            else:
                return f'{self.name} ({self.left})'
        else:
            return f'{self.name}'
    
    def __eq__(self, other):
        return self.type == other.type and self.name == other.name and self.rank == other.rank and self.left == other.left and self.right == other.right
    
    def __hash__(self): # Necessary for instances to behave sanely in dicts and sets.
        return hash((self.type, self.name, self.rank, self.left, self.right))

    def get_all_subtrees(self):
        if not self.left:
            leftSubtrees = [None]
        else:
            leftSubtrees = self.left.get_all_subtrees()
        if not self.right:
            rightSubtrees = [None]
        else:
            rightSubtrees = self.right.get_all_subtrees()
        return [self] + leftSubtrees + rightSubtrees
    
    def dot(self, filename):
        g = Digraph(format='png', strict=True, edge_attr={'dir': 'back'})
        nodes = self.get_all_subtrees()
        for node in nodes:
            if node:
                g.node(str(node.id), str(node.name) + '(' + str(node.rank) + ')')
        for node in nodes:
            if node and node.left:
                g.edge(str(node.id), str(node.left.id))
            if node and node.right:
                g.edge(str(node.id), str(node.right.id))
        g.render(filename)
