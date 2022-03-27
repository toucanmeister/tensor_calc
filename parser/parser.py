from copy import deepcopy
from typing import Dict
from scanner import *
from tree import *

class Parser():
    def __init__(self, input):
        self.tree = None    # Stores expression as a binary tree
        self.variable_ranks = {}  # Stores declared variables and their ranks
        self.scanner = Scanner(input)
        self.get_sym()
    
    def get_sym(self):
        (self.desc, self.ident) = self.scanner.get_sym()

    def fits(self, symbol):
        return self.desc == symbol
    
    def error(self, expected):
        raise Exception(f'Expected {expected} but found \'{self.ident}\'')

    def start(self):
        self.declaration()
        self.argument()
        self.tree = self.expressionpart()
    
    def declaration(self):
        if self.fits(TOKEN_ID.DECLARE):
            self.get_sym()
            self.tensordeclaration()
            while not self.fits(TOKEN_ID.ARGUMENT):
                self.tensordeclaration()
        else:
            self.error(TOKEN_ID.DECLARE.value)
    
    def tensordeclaration(self):
        if self.fits(TOKEN_ID.ALPHANUM) or self.fits(TOKEN_ID.LOWERCASE_ALPHA):
            variablename = self.ident
            self.get_sym()
        else:
            self.error('tensorname')
        if self.fits(TOKEN_ID.NATNUM):
            rank = int(self.ident)
            self.variable_ranks[variablename] = rank
            self.get_sym()
        else:
            self.error(TOKEN_ID.NATNUM.value)
    
    def argument(self):
        if self.fits(TOKEN_ID.ARGUMENT):
            self.get_sym()
            if self.fits(TOKEN_ID.ALPHANUM) or self.fits(TOKEN_ID.LOWERCASE_ALPHA):
                self.get_sym()
            else:
                self.error(TOKEN_ID.ALPHANUM.value)
        else:
            self.error(TOKEN_ID.ARGUMENT.value)
    
    def expressionpart(self):
        if self.fits(TOKEN_ID.EXPRESSION):
            self.get_sym()
            tree = self.expr()
        else:
            self.error(TOKEN_ID.EXPRESSION.value)
        return tree

    def expr(self):
        tree = self.term()
        while self.fits(TOKEN_ID.PLUS) or self.fits(TOKEN_ID.MINUS):
            type = NODETYPE.SUM if self.fits(TOKEN_ID.PLUS) else NODETYPE.DIFFERENCE
            name = '+' if self.fits(TOKEN_ID.PLUS) else '-'
            self.get_sym()
            tree = Tree(type, name, tree, self.term())
        return tree
    
    def term(self):
        tree = self.factor()
        while self.fits(TOKEN_ID.MULTIPLY):
            self.get_sym()
            if self.fits(TOKEN_ID.LRBRACKET):
                self.get_sym()
            else:
                self.error(TOKEN_ID.LRBRACKET.value)
            leftIndices, rightIndices, resultIndices = self.productindices()
            if self.fits(TOKEN_ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(TOKEN_ID.RRBRACKET.value)
            tree = Tree(NODETYPE.PRODUCT, f'*({leftIndices},{rightIndices}->{resultIndices})', tree, self.factor())
            tree.set_indices(leftIndices, rightIndices, resultIndices)
        return tree

    def productindices(self):
        leftIndices = ''
        rightIndices = ''
        resultIndices = ''
        while self.fits(TOKEN_ID.LOWERCASE_ALPHA):
            leftIndices = leftIndices + self.ident
            self.get_sym()
        if self.fits(TOKEN_ID.COMMA):
            self.get_sym()
        else:
            self.error(TOKEN_ID.COMMA.value)
        while self.fits(TOKEN_ID.LOWERCASE_ALPHA):
            rightIndices += self.ident
            self.get_sym()
        if self.fits(TOKEN_ID.MINUS):
            self.get_sym()
        else:
            self.error(TOKEN_ID.MINUS.value)
        if self.fits(TOKEN_ID.GREATER):
            self.get_sym()
        else:
            self.error(TOKEN_ID.GREATER.value)
        while self.fits(TOKEN_ID.LOWERCASE_ALPHA):
            resultIndices += self.ident
            self.get_sym()
        return leftIndices, rightIndices, resultIndices

    def factor(self):
        parity = 0
        while self.fits(TOKEN_ID.MINUS):
            parity = (parity+1) % 2
            self.get_sym()
        if parity == 1:
            tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, self.atom())
        else:
            tree = self.atom()
        while self.fits(TOKEN_ID.POW):
            self.get_sym()
            if self.fits(TOKEN_ID.LRBRACKET):
                self.get_sym()
                tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '^', tree, self.factor())
                if self.fits(TOKEN_ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(TOKEN_ID.RRBRACKET.value)
            else:
                tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '^', tree, self.atom())
        return tree
    
    def atom(self):
        if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
            tree = Tree(NODETYPE.CONSTANT, self.ident)
            self.get_sym()
        elif self.fits(TOKEN_ID.MINUS):
            self.get_sym()
            if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
                tree = Tree(NODETYPE.CONSTANT, '-' + self.ident)
                self.get_sym()
            else:
                self.error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.NATNUM.value)
        elif self.fits(TOKEN_ID.FUNCTION):
            functionName = self.ident
            self.get_sym()
            if self.fits(TOKEN_ID.LRBRACKET):
                self.get_sym()
                tree = Tree(NODETYPE.FUNCTION, functionName, None, self.expr())
                if self.fits(TOKEN_ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(TOKEN_ID.RRBRACKET.value)
            else:
                self.error(TOKEN_ID.LRBRACKET.value)
        elif self.fits(TOKEN_ID.ALPHANUM) or self.fits(TOKEN_ID.LOWERCASE_ALPHA):
            tree = Tree(NODETYPE.VARIABLE, self.ident)
            self.get_sym()
        elif self.fits(TOKEN_ID.LRBRACKET):
            self.get_sym()
            tree = self.expr()
            if self.fits(TOKEN_ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(TOKEN_ID.RRBRACKET.value)
        else:
            self.error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.FUNCTION.value + ' or ' + 'tensorname' +  ' or ' + TOKEN_ID.LRBRACKET.value)
        return tree

    def set_node_tensorrank(self):
        def set_tensorrank(node):
            if not node:
                return
            set_tensorrank(node.left)
            set_tensorrank(node.right)
            if node.type == NODETYPE.CONSTANT:
                node.rank = 0
            elif node.type == NODETYPE.VARIABLE:
                node.rank = self.variable_ranks[node.name]
            elif node.type == NODETYPE.ELEMENTWISE_FUNCTION:
                node.rank = node.right.rank
            elif node.type == NODETYPE.FUNCTION:
                pass
                #TODO: EACH FUNCTION NEEDS IT'S OWN IMPLEMENTATION HERE
            elif node.type in [NODETYPE.SUM, NODETYPE.DIFFERENCE, NODETYPE.QUOTIENT]:
                if node.right.rank != node.left.rank:
                    raise Exception(f'Ranks of inputs \'{node.left.name}\' ({node.left.rank}) and \'{node.right.name}\' ({node.right.rank}) to node of type {node.type.value} do not match.')
                else:
                    node.rank = node.left.rank
            elif node.type == NODETYPE.PRODUCT:
                if node.left.rank != len(node.leftIndices):
                    raise Exception(f'Rank of left input \'{node.left.name}\' ({node.left.rank}) to product node does not match product indices \'{node.leftIndices}\'.')
                elif node.right.rank != len(node.rightIndices):
                    raise Exception(f'Rank of right input \'{node.right.name}\' ({node.right.rank}) to product node does not match product indices \'{node.rightIndices}\'.')
                else:
                    node.rank = len(node.resultIndices)
            else:
                raise Exception(f'Unknown node type at node \'{node.name}\'.')
        set_tensorrank(self.tree)
        
    def eliminate_common_subtrees(self):
        subtrees = self.tree.get_all_subtrees()
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
        
            
                
        

if __name__ == '__main__':
    example = 'declare a 1 b 1 c 2 argument a expression a*(i,j->ij)b + a*(i,ij->ij)c + a*(i,j->ij)b + b*(i,ij->ij)(a*(i,j->ij)b)'
    p = Parser(example)
    p.start()
    p.set_node_tensorrank()
    p.tree.dot('tree')
    p.eliminate_common_subtrees()
    p.tree.dot('tree_cleaned')
