from scanner import *
from tree import *

class Parser():
    def __init__(self, input):
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
        if self.fits(ID.DECLARE):
            self.get_sym()
            self.tensordeclaration()
            while not self.fits(ID.ARGUMENT):
                self.tensordeclaration()
        else:
            self.error(ID.DECLARE.value)
    
    def tensordeclaration(self):
        if self.fits(ID.ALPHANUM) or self.fits(ID.LOWERCASE_ALPHA):
            self.get_sym()
        else:
            self.error('tensorname')
        if self.fits(ID.NATNUM):
            self.get_sym()
        else:
            self.error(ID.NATNUM.value)
    
    def argument(self):
        if self.fits(ID.ARGUMENT):
            self.get_sym()
            if self.fits(ID.ALPHANUM) or self.fits(ID.LOWERCASE_ALPHA):
                self.get_sym()
            else:
                self.error(ID.ALPHANUM.value)
        else:
            self.error(ID.ARGUMENT.value)
    
    def expressionpart(self):
        if self.fits(ID.EXPRESSION):
            self.get_sym()
            tree = self.expr()
        else:
            self.error(ID.EXPRESSION.value)
        return tree

    def expr(self):
        tree = self.term()
        while self.fits(ID.PLUS) or self.fits(ID.MINUS):
            op = '+' if self.fits(ID.PLUS) else '-'
            self.get_sym()
            tree = Tree(op, tree, self.term())
        return tree
    
    def term(self):
        tree = self.factor()
        while self.fits(ID.MULTIPLY):
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
            else:
                self.error(ID.LRBRACKET.value)
            leftIndices, rightIndices, resultIndices = self.productindices()
            if self.fits(ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(ID.RRBRACKET.value)
            tree = Tree('*', tree, self.factor())
            tree.set_indices(leftIndices, rightIndices, resultIndices)
        return tree

    def productindices(self):
        leftIndices = ''
        rightIndices = ''
        resultIndices = ''
        while self.fits(ID.LOWERCASE_ALPHA):
            leftIndices += self.ident
            self.get_sym()
        if self.fits(ID.COMMA):
            self.get_sym()
        else:
            self.error(ID.COMMA.value)
        while self.fits(ID.LOWERCASE_ALPHA):
            rightIndices += self.ident
            self.get_sym()
        if self.fits(ID.MINUS):
            self.get_sym()
        else:
            self.error(ID.MINUS.value)
        if self.fits(ID.GREATER):
            self.get_sym()
        else:
            self.error(ID.GREATER.value)
        while self.fits(ID.LOWERCASE_ALPHA):
            resultIndices += self.ident
            self.get_sym()
        return leftIndices, rightIndices, resultIndices

    def factor(self):
        parity = 0
        while self.fits(ID.MINUS):
            parity = (parity+1) % 2
            self.get_sym()
        if parity == 1:
            tree = Tree('-', None, self.atom())
        else:
            tree = self.atom()
        while self.fits(ID.POW):
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
                tree = Tree('^', tree, self.factor())
                if self.fits(ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(ID.RRBRACKET.value)
            else:
                tree = Tree('^', tree, self.atom())
        return tree
    
    def atom(self):
        if self.fits(ID.CONSTANT) or self.fits(ID.NATNUM):
            tree = Tree(self.ident)
            self.get_sym()
        elif self.fits(ID.MINUS):
            self.get_sym()
            if self.fits(ID.CONSTANT) or self.fits(ID.NATNUM):
                tree = Tree('-' + self.ident)
                self.get_sym()
            else:
                self.error(ID.CONSTANT.value + ' or ' + ID.NATNUM.value)
        elif self.fits(ID.FUNCTION):
            functionName = self.ident
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
                tree = Tree(functionName, None, self.expr())
                if self.fits(ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(ID.RRBRACKET.value)
            else:
                self.error(ID.LRBRACKET.value)
        elif self.fits(ID.ALPHANUM) or self.fits(ID.LOWERCASE_ALPHA):
            tree = Tree(self.ident)
            self.get_sym()
        elif self.fits(ID.LRBRACKET):
            self.get_sym()
            tree = self.expr()
            if self.fits(ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(ID.RRBRACKET.value)
        else:
            self.error(ID.CONSTANT.value + ' or ' + ID.FUNCTION.value + ' or ' + 'tensorname' +  ' or ' + ID.LRBRACKET.value)
        return tree


if __name__ == '__main__':
    example = 'declare a 1 b 1 argument a expression cos(a*(i,j->ij)b)'
    p = Parser(example)
    p.start()
    p.tree.dot('tree')
