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
        raise Exception(f'Expected \'{expected}\' but found \'{self.ident}\'')

    def input(self):
        self.declaration()
        self.argument()
        self.expressionpart()
    
    def declaration(self):
        if self.fits('declare'):
            self.get_sym()
            self.tensordeclaration()
        else:
            self.error('declare')