from enum import Enum
from unittest import result
from graphviz import Digraph
import string

from numpy import indices

class NODETYPE(Enum):
    CONSTANT = 'constant'
    VARIABLE = 'variable'
    SPECIAL_FUNCTION = 'function'
    ELEMENTWISE_FUNCTION = 'elementwise_function'
    SUM = 'sum'
    DIFFERENCE = 'difference'
    PRODUCT = 'product'
    POWER = 'power'
    DELTA = 'delta'

class Tree():
    node_counter = 0        # Running id for nodes, just used for the visualization
    axes_counter = 1        # Running id for axes
    axis_to_origin = {}     # Axis -> Name of node from which that axis originated, along with the index of that axis in the original node
    constant_counter = 0    # Running id for constants
    printing_constants = {} # A dict for saving constants that only get created during printing and their axes 
                            # (this is for convenience when transforming elementwise_inverse(x) to 1/x during printing)
    
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
        self.id = Tree.node_counter   # This is just for the visualization
        Tree.node_counter += 1
    
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
                    const = f'1_{Tree.new_constant()}'
                    Tree.printing_constants[const] = self.right.axes
                    return f'({const} / ({self.right}))'
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
                axis = Tree.new_axis()
                Tree.axis_to_origin[axis] = 'broadcast_axis_product' # These axes should get overriden by axes coming from variables
                desired_axes.append(axis)
            if not self.left.try_broadcasting(len(self.leftIndices), desired_axes):
                raise Exception(f'Rank of left input \'{self.left.name}\' ({self.left.rank}) to product node does not match product indices \'{self.leftIndices}\'.')
        elif self.right.rank != len(self.rightIndices):
            desired_axes = []
            for i in self.rightIndices:
                axis = Tree.new_axis()
                Tree.axis_to_origin[axis] = 'broadcast_axis_product' # These axes should get overriden by axes coming from variables
                desired_axes.append(axis)
            if not self.right.try_broadcasting(len(self.rightIndices), desired_axes):
                raise Exception(f'Rank of right input \'{self.right.name}\' ({self.right.rank}) to product node does not match product indices \'{self.rightIndices}\'.')
        for index in self.resultIndices:
            if not (index in self.leftIndices or index in self.rightIndices):
                raise Exception(f'Result index \'{index}\' of product node \'{self.name}\' not in left or right index set.')

    @classmethod
    def new_axis(cls):
        axis = Tree.axes_counter
        Tree.axes_counter += 1
        return axis

    @classmethod
    def new_constant(cls):
        constant = Tree.constant_counter
        Tree.constant_counter += 1
        return constant

    def set_tensorrank(self, variable_ranks, arg):
        if self.left:
            self.left.set_tensorrank(variable_ranks, arg)
        if self.right:
            self.right.set_tensorrank(variable_ranks, arg)
        if self.type == NODETYPE.CONSTANT or self.type == NODETYPE.DELTA:   # If we reach a constant, it keeps rank -1 and will get broadcasted later (unless it already has a rank)
            pass
        elif self.type == NODETYPE.VARIABLE:
            self.rank = variable_ranks[self.name]
            if self.axes == []:
                for i in range(self.rank):
                    axis = Tree.new_axis()
                    self.axes.append(axis)
                    Tree.axis_to_origin[axis] = f'{self.name}[{i}]'  # All axes should originally come from a variable
        elif self.type == NODETYPE.ELEMENTWISE_FUNCTION:
            self.rank = self.right.rank
            self.axes = self.right.axes
        elif self.type == NODETYPE.SPECIAL_FUNCTION:
            if self.name == 'inv':
                if self.right.rank != 2:
                    axis = Tree.new_axis()
                    Tree.axis_to_origin[axis] = 'broadcast_axis_inv' # These axes should get overriden by axes coming from variables
                    if not self.right.try_broadcasting(2, [axis, axis]):
                        raise Exception(f'Rank of operand \'{self.right}\' to inv node is not 2.')
                self.rank = 2
                self.axes = self.right.axes
            elif self.name == 'det':
                if self.right.rank != 2:
                    axis = Tree.new_axis()
                    Tree.axis_to_origin[axis] = 'broadcast_axis_det' # These axes should get overriden by axes coming from variables
                    if not self.right.try_broadcasting(2, [axis, axis]):
                        raise Exception(f'Rank of operand \'{self.right}\' to inv node is not 2.')
                self.rank = 0
                self.axes = []
            elif self.name == 'adj':
                if self.right.rank != 2:
                    axis = Tree.new_axis()
                    Tree.axis_to_origin[axis] = 'broadcast_axis_adj' # These axes should get overriden by axes coming from variables
                    if not self.right.try_broadcasting(2, [axis, axis]):
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
            for i in range(len(self.axes)):
                if Tree.axis_to_origin[self.axes[i]].startswith('broadcast_axis'): # This axis does not come from a variable, needs to be overriden
                    self.axes[i] = self.right.axes[i]
        elif self.type == NODETYPE.POWER:
            if self.right.rank != 0:
                if not self.right.try_broadcasting(0, []):
                    raise Exception(f'Rank of right operand \'{self.right.name}\' ({self.right.rank}) in power node is not 0.')
            self.rank = self.left.rank
            self.axes = self.left.axes
        elif self.type == NODETYPE.PRODUCT:
            if self.name == '_TO_BE_SET_ELEMENTWISE': # Turn this into an elementwise product
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
                    if i in self.leftIndices and i in self.rightIndices: # If we may take it from both left and right, choose the one with the axis coming from a variable
                        left_axis = self.left.axes[self.leftIndices.index(i)]
                        right_axis = self.right.axes[self.rightIndices.index(i)]
                        if Tree.axis_to_origin[left_axis].startswith('broadcast_axis'):
                            self.axes.append(right_axis)
                        else:
                            self.axes.append(left_axis)
                    elif i in self.leftIndices:
                        self.axes.append(self.left.axes[self.leftIndices.index(i)])
                    elif i in self.rightIndices:
                        self.axes.append(self.right.axes[self.rightIndices.index(i)])
        else:
            raise Exception(f'Unknown node type at node \'{self.name}\'.')

    def unify_axes(self):
        if self.type == NODETYPE.CONSTANT or self.type == NODETYPE.DELTA:
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
            for i in range(len(self.resultIndices)): # Sets left and right child axes to the axes in this node, using indices as guidance
                for j in range(len(self.leftIndices)):
                    if self.resultIndices[i] == self.leftIndices[j]:
                        self.left.axes[j] = self.axes[i]
                for j in range(len(self.rightIndices)):
                    if self.resultIndices[i] == self.rightIndices[j]:
                        self.right.axes[j] = self.axes[i]
            for i in range(len(self.leftIndices)): # Unifies left and right child axes that are the same, using indices as guidance
                indexInRight = self.rightIndices.find(self.leftIndices[i])
                if indexInRight != -1:
                    left_axis = self.left.axes[i]
                    right_axis = self.right.axes[indexInRight]
                    if Tree.axis_to_origin[left_axis].startswith('broadcast_axis'): # Axis that does not come from a variable, we should choose the other one
                        self.get_root().rename_axis(left_axis, right_axis)
                    else:
                        self.get_root().rename_axis(right_axis, left_axis)
        if self.left:
            self.left.unify_axes()
        if self.right:
            self.right.unify_axes()
    
    def rename_axis(self, axis_to_rename, new_name):
        if axis_to_rename in Tree.axis_to_origin.keys() and new_name in Tree.axis_to_origin.keys() and axis_to_rename != new_name:
            Tree.axis_to_origin.pop(axis_to_rename)
        elif axis_to_rename in Tree.axis_to_origin.keys() and new_name not in Tree.axis_to_origin.keys():
            Tree.axis_to_origin[new_name] = Tree.axis_to_origin[axis_to_rename]
            Tree.axis_to_origin.pop(axis_to_rename)
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
        if self and self.left:
            self.left.incoming.append(self)
            self.left.add_incoming_edges()
        if self and self.right:
            self.right.incoming.append(self)
            self.right.add_incoming_edges()
        
    def remove_nonexistant_axes(self):  # Removes from Tree.axis_to_origin all axes that do not occur in this (sub)tree
        def contains_axis(node, axis):
            if not node:
                return False
            else:
                return (axis in node.axes) or contains_axis(node.left, axis) or contains_axis(node.right, axis)
        axes_to_remove = []
        for axis in Tree.axis_to_origin.keys():
            if not contains_axis(self, axis):
                axes_to_remove.append(axis)
        for axis in axes_to_remove:
            Tree.axis_to_origin.pop(axis)
        
    def remove_unneccessary_deltas(self):
        def left_indices_fit():
            is_delta_zero = self.leftIndices == '' and self.rightIndices == self.resultIndices
            fits_one_way = self.rightIndices == self.leftIndices[:(len(self.leftIndices)//2)] and self.resultIndices == self.leftIndices[(len(self.leftIndices)//2):]
            fits_the_other_way = self.rightIndices == self.leftIndices[(len(self.leftIndices)//2):] and self.resultIndices == self.leftIndices[:(len(self.leftIndices)//2)]
            return is_delta_zero or fits_one_way or fits_the_other_way
        def right_indices_fit():
            is_delta_zero = self.rightIndices == '' and self.leftIndices == self.resultIndices
            fits_one_way = self.leftIndices == self.rightIndices[:(len(self.rightIndices)//2)] and self.resultIndices == self.rightIndices[(len(self.rightIndices)//2):]
            fits_the_other_way = self.leftIndices == self.rightIndices[(len(self.rightIndices)//2):] and self.resultIndices == self.rightIndices[:(len(self.rightIndices)//2)]
            return is_delta_zero or fits_one_way or fits_the_other_way
        new_self = self
        if self.type == NODETYPE.PRODUCT:
            if self.left.name.startswith('_delta') and left_indices_fit():
                new_self = self.right
                self.add_incoming_edges()
                for parent in self.incoming:
                    if parent.left == self:
                        parent.left = new_self
                    if parent.right == self:
                        parent.right = new_self
            if self.right.name.startswith('_delta') and right_indices_fit():
                new_self = self.left
                self.add_incoming_edges()
                for parent in self.incoming:
                    if parent.left == self:
                        parent.left = new_self
                    if parent.right == self:
                        parent.right = new_self
        if self.left:
            self.left.remove_unneccessary_deltas()
        if self.right:
            self.right.remove_unneccessary_deltas()
        return new_self

    def fix_missing_indices(self, arg):
        def add_blowup(resultIndices, missingIndices):
            blowup = Tree(NODETYPE.PRODUCT, f'*({resultIndices+missingIndices},{missingIndices}->{resultIndices})', self, Tree(NODETYPE.CONSTANT, f'1_{Tree.new_constant()}'))
            blowup.set_indices(resultIndices+missingIndices, missingIndices, resultIndices)
            self.add_incoming_edges()
            for parent in self.incoming:
                if parent.left == self:
                    parent.left = blowup
                if parent.right == self:
                    parent.right = blowup
            return blowup
        new_self = self
        if self.type == NODETYPE.PRODUCT:
            s1 = self.leftIndices
            s2 = self.rightIndices
            s3 = self.resultIndices
            missingIndices = ''
            if self.left and self.left.contains(arg):
                missingIndices = ''.join([index for index in s1 if not (index in s2 or index in s3)])
            elif self.right and self.right.contains(arg):
                missingIndices = ''.join([index for index in s2 if not (index in s1 or index in s3)])

            if missingIndices: # Special problem case
                oldResultIndices = self.resultIndices
                self.resultIndices += missingIndices
                self.name = f'*({s1},{s2}->{self.resultIndices})'
                new_self = add_blowup(oldResultIndices, missingIndices)
                
            if self.left:
                self.left.fix_missing_indices(arg)
            if self.right:
                self.right.fix_missing_indices(arg)
        return new_self
    
    def rename_equivalent_constants(self):
        constants = []
        def helper(node):
            if node.type == NODETYPE.CONSTANT:
                equivalent_constant_in_list = None
                for constant in constants:
                    if constant.name.split('_')[0] == node.name.split('_')[0] and constant.axes == node.axes: # Same number and same axes
                        equivalent_constant_in_list = constant
                if node not in constants and not equivalent_constant_in_list:
                    constants.append(node)
                elif node not in constants and equivalent_constant_in_list:
                    node.name = constant.name
            if node.left:
                helper(node.left)
            if node.right:
                helper(node.right)
        helper(self)