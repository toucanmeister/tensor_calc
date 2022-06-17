from enum import Enum
from unittest import result
from graphviz import Digraph
import string

class NODETYPE(Enum):
    CONSTANT = 'constant'
    VARIABLE = 'variable'
    SPECIAL_FUNCTION = 'function'
    ELEMENTWISE_FUNCTION = 'elementwise_function'
    SUM = 'sum'
    DIFFERENCE = 'difference'
    PRODUCT = 'product'
    POWER = 'power'

class Tree():
    running_id = 0
    axes_counter = 1
    def __init__(self, nodetype, name, left=None, right=None):
        self.type = nodetype
        self.name = name
        self.left = left
        self.right = right
        self.incoming = []  # Nodes from incoming edges, only filled after common subexpression elimination
        self.rank = -1      # Initialization value
        self.axes = []
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
        if self.right:
            if self.left:
                return f'({self.left} {self.name} {self.right})'
            else:
                if self.name == 'elementwise_inverse':
                    return f'(1 / ({self.right}))'
                return f'({self.name}({self.right}))'
        else:
            return f'{self.name}'
    
    def __eq__(self, other):
        return other and self.type == other.type and self.name == other.name and self.rank == other.rank and self.left == other.left and self.right == other.right and self.axes == other.axes
    
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
    
    def dot(self, filename, print_axes=True):
        g = Digraph(format='png', edge_attr={'dir': 'back'}, graph_attr={'dpi': '300'})
        done_nodes = []
        nodes = self.get_all_subtrees()
        for node in nodes:
            if node:
                if print_axes:
                    g.node(str(node.id), f'{node.name} \n {node.axes}')
                else:
                    g.node(str(node.id), str(node.name))
        for node in nodes:
            if node not in done_nodes:
                if node and node.left:
                    g.edge(str(node.id), str(node.left.id), '<')
                if node and node.right:
                    g.edge(str(node.id), str(node.right.id), '>')
                done_nodes.append(node)
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
            desired_axes = []
            for i in self.leftIndices:
                desired_axes.append(self.new_axis())
            if not self.left.try_broadcasting(len(self.leftIndices), desired_axes):
                raise Exception(f'Rank of left input \'{self.left.name}\' ({self.left.rank}) to product node does not match product indices \'{self.leftIndices}\'.')
        elif self.right.rank != len(self.rightIndices):
            desired_axes = []
            for i in self.rightIndices:
                desired_axes.append(self.new_axis())
            if not self.right.try_broadcasting(len(self.rightIndices), desired_axes):
                raise Exception(f'Rank of right input \'{self.right.name}\' ({self.right.rank}) to product node does not match product indices \'{self.rightIndices}\'.')
        for index in self.resultIndices:
            if not (index in self.leftIndices or index in self.rightIndices):
                raise Exception(f'Result index \'{index}\' of product node \'{self.name}\' not in left or right index set.')

    def new_axis(self):
        axis = Tree.axes_counter
        Tree.axes_counter += 1
        return axis

    def set_tensorrank(self, variable_ranks, arg):
        if self.left:
            self.left.set_tensorrank(variable_ranks, arg)
        if self.right:
            self.right.set_tensorrank(variable_ranks, arg)
        if self.type == NODETYPE.CONSTANT:   # If we reach a constant, it gets rank -1, and will get broadcasted later.
            self.rank = -1
        elif self.type == NODETYPE.VARIABLE:
            self.rank = variable_ranks[self.name]
            if self.axes == []:
                for i in range(self.rank):
                    self.axes.append(self.new_axis())
        elif self.type == NODETYPE.ELEMENTWISE_FUNCTION:
            self.rank = self.right.rank
            self.axes = self.right.axes
        elif self.type == NODETYPE.SPECIAL_FUNCTION:
            if self.name == 'inv':
                if self.right.rank != 2:
                    if not self.right.try_broadcasting(2, [self.new_axis(), self.new_axis()]):
                        raise Exception(f'Rank of operand \'{self.right}\' to inv node is not 2.')
                self.rank = 2
                self.axes = self.right.axes
            elif self.name == 'det':
                if self.right.rank != 2:
                    if not self.right.try_broadcasting(2, [self.new_axis(), self.new_axis()]):
                        raise Exception(f'Rank of operand \'{self.right}\' to inv node is not 2.')
                self.rank = 0
                self.axes = []
            elif self.name == 'adj':
                if self.right.rank != 2:
                    if not self.right.try_broadcasting(2, [self.new_axis(), self.new_axis()]):
                        raise Exception(f'Rank of operand \'{self.right}\' to inv node is not 2.')
                self.rank = 2
                self.axes = self.right.axes
        elif self.type == NODETYPE.SUM or self.type == NODETYPE.DIFFERENCE:
            if self.right.rank != self.left.rank:
                if not self.left.try_broadcasting(self.right.rank, self.right.axes):
                    if not self.right.try_broadcasting(self.left.rank, self.left.axes):
                        raise Exception(f'Ranks of operands \'{self.left.name}\' ({self.left.rank}) and \'{self.right.name}\' ({self.right.rank}) in sum node do not match.')
            self.rank = self.left.rank
            self.axes = self.left.axes
        elif self.type == NODETYPE.POWER:
            if self.right.rank != 0:
                if not self.right.try_broadcasting(0, []):
                    raise Exception(f'Rank of right operand \'{self.right.name}\' ({self.right.rank}) in power node is not 0.')
            self.rank = self.left.rank
            self.axes = self.left.axes
        elif self.type == NODETYPE.PRODUCT:
            if self.name == '_TO_BE_SET_ELEMENTWISE':
                if self.left.rank == -1:
                    indices = ''.join([i for i in string.ascii_lowercase][0:self.right.rank])
                else:
                    indices = ''.join([i for i in string.ascii_lowercase][0:self.left.rank])
                self.set_indices(indices, indices, indices)
                self.name = f'*({indices},{indices}->{indices})'
            self.check_multiplication()
            self.rank = len(self.resultIndices)
            if self.axes == []:
                for i in self.resultIndices: # Gets the correct axes from left and right child nodes
                    if i in self.leftIndices:
                        self.axes.append(self.left.axes[self.leftIndices.index(i)])
                    elif i in self.rightIndices:
                        self.axes.append(self.right.axes[self.rightIndices.index(i)])
        else:
            raise Exception(f'Unknown node type at node \'{self.name}\'.')

    def unify_axes(self):
        if self.type == NODETYPE.CONSTANT:
            pass
        if self.type == NODETYPE.VARIABLE:
            pass
        if self.type == NODETYPE.ELEMENTWISE_FUNCTION:
            self.right.axes = self.axes
        if self.type == NODETYPE.SPECIAL_FUNCTION:
            if self.name == 'inv' or self.name == 'adj' or self.name =='det':
                old_axis = self.right.axes[1]
                self.right.axes[1] = self.right.axes[0]
                self.get_root().rename_axis(old_axis, self.right.axes[0])
            if self.name == 'inv' or self.name == 'adj':
                self.right.axes = self.axes
        if self.type == NODETYPE.SUM or self.type == NODETYPE.DIFFERENCE:
            self.left.axes = self.axes
            self.right.axes = self.axes
        if self.type == NODETYPE.POWER:
            self.left.axes = self.axes
        if self.type == NODETYPE.PRODUCT:
            for i in range(len(self.resultIndices)):
                for j in range(len(self.leftIndices)):
                    if self.resultIndices[i] == self.leftIndices[j]:
                        self.left.axes[j] = self.axes[i]
                for j in range(len(self.rightIndices)):
                    if self.resultIndices[i] == self.rightIndices[j]:
                        self.right.axes[j] = self.axes[i]
        if self.left:
            self.left.unify_axes()
        if self.right:
            self.right.unify_axes()
    
    def rename_axis(self, axis_to_rename, new_name):
        self.axes = [new_name if axis == axis_to_rename else axis for axis in self.axes]
        if self.left:
            self.left.rename_axis(axis_to_rename, new_name)
        if self.right:
            self.right.rename_axis(axis_to_rename, new_name)
    
    def get_root(self): # Requires incoming edges which are added during CSE
        if not self.incoming:
            return self
        else:
            return self.incoming[0].get_root()


    def try_broadcasting(self, desired_rank, desired_axes):
        if self.type == NODETYPE.CONSTANT:
            self.rank = desired_rank
            self.axes = desired_axes
            return True
        elif self.type == NODETYPE.ELEMENTWISE_FUNCTION:
            if self.right.try_broadcasting(desired_rank, desired_axes):
                self.rank = desired_rank
                self.axes = desired_axes
                return True
            else:
                return False
        else:
            return False

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
        if self and self.left and (self not in self.left.incoming):
            self.left.incoming.append(self)
            self.left.add_incoming_edges()
        if self and self.right and (self not in self.right.incoming):
            self.right.incoming.append(self)
            self.right.add_incoming_edges()