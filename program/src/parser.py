from scanner import *
from tree import *

# Important convention: Unary functions have their argument in 'right' and None in 'left'

class Parser():
    def __init__(self, input):
        self.dag = None    # Stores expression as a binary tree (Until CSE, then it's a binary DAG)
        self.variable_ranks = {}  # Stores declared variables and their ranks
        self.arg_name = None  # Stores the derivation argument variable name
        self.scanner = Scanner(input)
        self.get_sym()
    
    def get_sym(self):
        (self.desc, self.ident) = self.scanner.get_sym()

    def fits(self, symbol):
        return self.desc == symbol
    
    def error(self, expected):
        raise Exception(f'Expected {expected} but found \'{self.ident}\'')

    def parse(self):
        self.declaration()
        self.argument()
        self.dag = self.expressionpart()
        self.dag.eliminate_common_subtrees()
        self.dag.set_tensorrank(self.variable_ranks)
    
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
                self.arg_name = self.ident
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
            if self.fits(TOKEN_ID.PLUS):
                self.get_sym()
                tree = Tree(NODETYPE.SUM, '+', tree, self.term())
            else:
                self.get_sym()
                tree = Tree(NODETYPE.SUM, '+', tree, Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, self.term()))
        return tree
    
    def term(self):
        tree = self.factor()
        while self.fits(TOKEN_ID.MULTIPLY) or self.fits(TOKEN_ID.DIVIDE):
            if self.fits(TOKEN_ID.MULTIPLY):
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
            if self.fits(TOKEN_ID.DIVIDE):
                self.get_sym()
                tree = Tree(NODETYPE.QUOTIENT, '/', tree, self.factor())
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
                tree = Tree(NODETYPE.POWER, '^', tree, self.factor())
                if self.fits(TOKEN_ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(TOKEN_ID.RRBRACKET.value)
            else:
                tree = Tree(NODETYPE.POWER, '^', tree, self.atom())
        return tree
    
    def atom(self):
        if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
            tree = Tree(NODETYPE.CONSTANT, self.ident)
            self.get_sym()
        elif self.fits(TOKEN_ID.MINUS):
            self.get_sym()
            if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
                tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.CONSTANT, self.ident))
                self.get_sym()
            else:
                self.error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.NATNUM.value)
        elif self.fits(TOKEN_ID.FUNCTION):
            functionName = self.ident
            self.get_sym()
            if self.fits(TOKEN_ID.LRBRACKET):
                self.get_sym()
                tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, functionName, None, self.expr())
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

        
if __name__ == '__main__':
    example = 'declare a 1 b 1 argument a expression a - b'
    p = Parser(example)
    p.parse()
    p.dag.set_tensorrank(p.variable_ranks)
    p.dag.dot('tree')
    p.dag.eliminate_common_subtrees()
    p.dag.add_incoming_edges()
    p.dag.dot('tree_cleaned')
    print(p.dag)