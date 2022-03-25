from asyncio import constants
from graphviz import Digraph
from enum import Enum

class NODETYPE(Enum):
    CONSTANT = 'constant'
    VARIABLE = 'variable'
    FUNCTION = 'function'
    ELEMENTWISE_FUNCTION = 'elementwise_function'
    SUM = 'sum'
    DIFFERENCE = 'difference'
    PRODUCT = 'product'
    QUOTIENT = 'quotient'


class Tree():
    def __init__(self, nodetype, name, left=None, right=None):
        self.type = type
        self.name = name
        self.left = left
        self.right = right
        self.rank = -1      # Initialization value
        if nodetype == NODETYPE.PRODUCT:    
            self.leftIndices = ''       # Indices in the einstein product notation
            self.rightIndices = ''
            self.resultIndices = ''
    
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
    
    def get_nodes(self):
        nodes = [self]
        if self.left:
            for node in self.left.get_nodes():
                nodes.append(node)
        if self.right:
            for node in self.right.get_nodes():
                nodes.append(node)
        return nodes
    
    def dot(self, filename, view=False):
        dot = Digraph(format='pdf')
        nodes_to_number = { }
        i = 0
        for node in self.get_nodes():
            nodes_to_number[node] = str(i)
            dot.node(str(i) , str(node.name))
            i += 1
        for node in self.get_nodes():
            if node.left:
                dot.edge(nodes_to_number[node], nodes_to_number[node.left])
            if node.right:
                dot.edge(nodes_to_number[node], nodes_to_number[node.right])
        dot.render(filename, view=view)
        return dot
