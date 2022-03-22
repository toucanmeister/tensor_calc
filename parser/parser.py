from scanner import *

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
        self.expressionpart()
    
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
            self.expr()
        else:
            self.error(ID.EXPRESSION.value)

    def expr(self):
        self.term()
        while self.fits(ID.PLUS) or self.fits(ID.MINUS):
            self.get_sym()
            self.term()
    
    def term(self):
        while self.fits(ID.MINUS):
            self.get_sym()
        self.factor()
        while self.fits(ID.MULTIPLY):
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
            else:
                self.error(ID.LRBRACKET.value)
            self.productindices()
            if self.fits(ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(ID.RRBRACKET.value)
            while self.fits(ID.MINUS):
                self.get_sym()
            self.factor()
    
    def productindices(self):
        if self.fits(ID.LOWERCASE_ALPHA):
            self.get_sym()
        else:
            self.error(ID.LOWERCASE_ALPHA.value)
        if self.fits(ID.COMMA):
            self.get_sym()
        else:
            self.error(ID.COMMA.value)
        if self.fits(ID.LOWERCASE_ALPHA):
            self.get_sym()
        else:
            self.error(ID.LOWERCASE_ALPHA.value)
        if self.fits(ID.MINUS):
            self.get_sym()
        else:
            self.error(ID.MINUS.value)
        if self.fits(ID.GREATER):
            self.get_sym()
        else:
            self.error(ID.GREATER.value)
        if self.fits(ID.LOWERCASE_ALPHA):
            self.get_sym()
        else:
            self.error(ID.LOWERCASE_ALPHA.value)

    def factor(self):
        self.atom()
        while self.fits(ID.POW):
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
                self.factor()
                if self.fits(ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(ID.RRBRACKET.value)
            else:
                self.atom()
    
    def atom(self):
        if self.fits(ID.CONSTANT) or self.fits(ID.NATNUM):
            self.get_sym()
        elif self.fits(ID.MINUS):
            self.get_sym()
            if self.fits(ID.CONSTANT) or self.fits(ID.NATNUM):
                self.get_sym()
            else:
                self.error(ID.CONSTANT.value + ' or ' + ID.NATNUM.value)
        elif self.fits(ID.FUNCTION):
            self.get_sym()
            if self.fits(ID.LRBRACKET):
                self.get_sym()
                self.expr()
                if self.fits(ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(ID.RRBRACKET.value)
            else:
                self.error(ID.LRBRACKET.value)
        elif self.fits(ID.ALPHANUM) or self.fits(ID.LOWERCASE_ALPHA):
            self.get_sym()
        elif self.fits(ID.LRBRACKET):
            self.get_sym()
            self.expr()
            if self.fits(ID.RRBRACKET):
                self.get_sym()
            else:
                self.error(ID.RRBRACKET.value)
        else:
            self.error(ID.CONSTANT.value + ' or ' + ID.FUNCTION.value + ' or ' + 'tensorname' +  ' or ' + ID.LRBRACKET.value)


if __name__ == '__main__':
    example = 'declare a0 1 b1 2 c 0 \n argument \t a0 \n expression a0*(ij,jk->ik)b1 + c'
    p = Parser(example)
    p.start()
