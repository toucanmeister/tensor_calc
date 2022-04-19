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
                return f'({self.left} {self.name} {self.right})'
            else:
                return f'{self.name} ({self.left})'
        else:
            return f'{self.name}'
    
    def __eq__(self, other):
        return other and self.type == other.type and self.name == other.name and self.rank == other.rank and self.left == other.left and self.right == other.right
    
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

    def contains(self, node):
        if self == node:
            return True
        else:
            return (self.left and self.left.contains(node)) or (self.right and self.right.contains(node))
    
    def find(self, nodename):
        if self.name == nodename:
            return self
        elif self.left and self.left.find(nodename):
            return self.left.find(nodename)
        elif self.right and self.right.find(nodename):
            return self.right.find(nodename)
        else:
            return None

    def check_multiplication(self):
        if self.left.rank != len(self.leftIndices):
                raise Exception(f'Rank of left input \'{self.left.name}\' ({self.left.rank}) to product node does not match product indices \'{self.leftIndices}\'.')
        elif self.right.rank != len(self.rightIndices):
                raise Exception(f'Rank of right input \'{self.right.name}\' ({self.right.rank}) to product node does not match product indices \'{self.rightIndices}\'.')
        for index in self.resultIndices:
            if not (index in self.leftIndices or index in self.rightIndices):
                raise Exception(f'Result index \'{index}\' of product node \'{self.name}\' not in left or right index set.')

    def set_tensorrank(self, variable_ranks):
        if self.left:
            self.left.set_tensorrank(variable_ranks)
        if self.right:
            self.right.set_tensorrank(variable_ranks)
        if self.type == NODETYPE.CONSTANT:
            self.rank = 0
        elif self.type == NODETYPE.VARIABLE:
            self.rank = variable_ranks[self.name]
        elif self.type == NODETYPE.ELEMENTWISE_FUNCTION:
            self.rank = self.right.rank
        elif self.type == NODETYPE.FUNCTION:
            pass
            #TODO: EACH FUNCTION NEEDS IT'S OWN IMPLEMENTATION HERE
        elif self.type in [NODETYPE.SUM, NODETYPE.DIFFERENCE, NODETYPE.QUOTIENT]:
            if self.right.rank != self.left.rank:
                raise Exception(f'Ranks of inputs \'{self.left.name}\' ({self.left.rank}) and \'{self.right.name}\' ({self.right.rank}) to node of type {self.type.value} do not match.')
            else:
                self.rank = self.left.rank
        elif self.type == NODETYPE.PRODUCT:
            self.check_multiplication()
            self.rank = len(self.resultIndices)
        else:
            raise Exception(f'Unknown node type at node \'{self.name}\'.')
    
    def eliminate_common_subtrees(self):
        subtrees = self.get_all_subtrees()
        hashmap = {}
        def helper(subtree):
            if not subtree: return
            if subtree.left in hashmap:
                subtree.left = hashmap[subtree.left]
            else:
                hashmap[subtree.left] = subtree.left
            if subtree.right in hashmap:
                subtree.right = hashmap[subtree.right]
            else:
                hashmap[subtree.right] = subtree.right
        for subtree in subtrees:
            helper(subtree)
    
    def add_incoming_edges(self):
        if self and self.left:
            self.left.incoming.append(self)
            self.left.add_incoming_edges()
        if self and self.right:
            self.right.incoming.append(self)
            self.right.add_incoming_edges()