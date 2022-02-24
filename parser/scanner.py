import string
from enum import Enum

class Input():
    def __init__(self, input):
        self.index = 0
        self.length = len(input)
        self.input = input
        
    def next(self):
        if self.index < self.length:
            literal = self.input[self.index]
            self.index += 1
            return literal
        else:
            return None

class ID(Enum):
    CONSTANT = 'constant'
    DECLARE = 'declare'
    ARGUMENT = 'argument'
    EXPRESSION = 'expression'
    FUNCTION = 'function'
    LOWERCASE_ALPHA = 'lowercase_alpha'
    ALPHANUM = 'alphanum'
    NONE = 'none'
    PLUS = 'plus'
    MINUS = 'minus'
    MULTIPLY = 'multiply'
    DIVIDE = 'divide'
    POW = 'pow'
    LRBRACKET = 'lrbracket'
    RRBRACKET = 'rrbracket'
    GREATER = 'greater'
    COMMA = 'comma'

ALPHA = list(string.ascii_letters)
DIGITS = [str(i) for i in range(10)]
SYMBOLS = {
    '+': ID.PLUS,
    '-': ID.MINUS,
    '*': ID.MULTIPLY,
    '/': ID.DIVIDE,
    '^': ID.POW,
    '(': ID.LRBRACKET,
    ')': ID.RRBRACKET,
    '>': ID.GREATER,
    ',': ID.COMMA
}
KEYWORDS = {
    'declare': ID.DECLARE,
    'argument': ID.ARGUMENT,
    'expression': ID.EXPRESSION
}
FUNCTIONS = {'sin', 'cos', 'exp', 'log', 'norm2', 'tr', 'det', 'logdet', 'inv', 'sqrt', 'abs', 'diag'}

class Scanner():
    def __init__(self, input):
        self.input = Input(input)
        self.current = self.input.next()
    
    def get_sym(self):
        identifier = ''

        self.ignore_whitespace()
        
        # CONSTANTS
        if self.current in DIGITS:
            id = ID.CONSTANT
            while self.current in DIGITS:
                identifier += self.read_and_shift()
            if self.current == ".":
                identifier += self.read_and_shift()
                if self.current in DIGITS:
                    while self.current in DIGITS:
                        identifier += self.read_and_shift()
                else:
                    raise Exception(f'Error: Expected digit after \'.\' in constant, but found \'{self.current}\'')
            if self.current in ['e', 'E']:
                identifier += 'e'
                self.current = self.input.next()
                if self.current in ['+', '-']:
                    identifier += self.read_and_shift()
                if self.current in DIGITS:
                    while self.current in DIGITS:
                        identifier += self.read_and_shift()
                else:
                    raise Exception(f'Error: Expected digit after \'e\'/\'E\' in constant, but found \'{self.current}\'')
        # SYMBOLS
        elif self.current in SYMBOLS.keys():
            id = SYMBOLS.get(self.current)
            identifier = self.current
            self.current = self.input.next()
        # KEYWORDS, FUNCTIONS, LOWERCASE_ALPHA AND OTHER WORDS
        elif self.current in ALPHA:
            while self.current in ALPHA or self.current in DIGITS:
                identifier += self.current
                self.current = self.input.next()
            if identifier.lower() in KEYWORDS.keys():
                id = KEYWORDS.get(identifier.lower())
                identifier = identifier.lower()
            elif identifier.lower() in FUNCTIONS:
                id = ID.FUNCTION
                identifier = identifier.lower()
            elif identifier.islower() and identifier.isalpha():
                id = ID.LOWERCASE_ALPHA
            else:
                id = ID.ALPHANUM
        # EMPTY
        elif self.current == None:
            id = ID.NONE
            identifier = None
        # UNKNOWN SYMBOL
        else:
            raise Exception(f'Symbol {self.current} not allowed.')
        return id, identifier

    
    def ignore_whitespace(self):
        while self.current != None and self.current in string.whitespace:
                self.current = self.input.next()
    
    def read_and_shift(self):
        identifier = self.current
        self.current = self.input.next()
        return identifier

if __name__ == '__main__':
    example = 'declare a0(ij) b1(kl) c(mn) \n argument\ta0 \n expression a0*(ij,jk->ik)b1 + c*cOs(2) - EXP(3.13)*Log(1e-1) * sin(75.003E2)'
    s = Scanner(example)
    desc, ident = s.get_sym()
    while ident:
        print(f'{desc} {ident}')
        desc, ident = s.get_sym()
