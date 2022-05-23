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

class TOKEN_ID(Enum):
    CONSTANT = 'constant'
    NATNUM = 'natnum'
    DECLARE = 'declare'
    DERIVATIVE = 'derivative'
    WRT = 'wrt'
    EXPRESSION = 'expression'
    SPECIAL_FUNCTION = 'special_function'
    ELEMENTWISE_FUNCTION = 'elementwise_function'
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
    '+': TOKEN_ID.PLUS,
    '-': TOKEN_ID.MINUS,
    '*': TOKEN_ID.MULTIPLY,
    '/': TOKEN_ID.DIVIDE,
    '^': TOKEN_ID.POW,
    '(': TOKEN_ID.LRBRACKET,
    ')': TOKEN_ID.RRBRACKET,
    '>': TOKEN_ID.GREATER,
    ',': TOKEN_ID.COMMA
}
KEYWORDS = {
    'declare': TOKEN_ID.DECLARE,
    'derivative': TOKEN_ID.DERIVATIVE,
    'expression': TOKEN_ID.EXPRESSION,
    'wrt': TOKEN_ID.WRT
}

ELEMENTWISE_FUNCTIONS = {'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'tanh', 'exp', 'log', 'abs', 'sign', 'relu'}
SPECIAL_FUNCTIONS = {'inv', 'det', 'adj'}

class Scanner():
    def __init__(self, input):
        self.input = Input(input)
        self.current = self.input.next()
    
    def get_sym(self):
        identifier = ''

        self.ignore_whitespace()
        
        # CONSTANTS AND NATNUMS
        if self.current in DIGITS:
            id = TOKEN_ID.NATNUM
            while self.current in DIGITS:
                identifier += self.read_and_shift()
            if self.current == ".":
                id = TOKEN_ID.CONSTANT
                identifier += self.read_and_shift()
                if self.current in DIGITS:
                    while self.current in DIGITS:
                        identifier += self.read_and_shift()
                else:
                    raise Exception(f'Error: Expected digit after \'.\' in constant, but found \'{self.current}\'')
            if self.current in ['e', 'E']:
                id = TOKEN_ID.CONSTANT
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
        # KEYWORDS, ELEMENTWISE_FUNCTIONS, SPECIAL_FUNCTIONS, LOWERCASE_ALPHA AND OTHER WORDS
        elif self.current in ALPHA:
            while self.current in ALPHA or self.current in DIGITS:
                identifier += self.current
                self.current = self.input.next()
            if identifier.lower() in KEYWORDS.keys():
                id = KEYWORDS.get(identifier.lower())
                identifier = identifier.lower()
            elif identifier.lower() in ELEMENTWISE_FUNCTIONS:
                id = TOKEN_ID.ELEMENTWISE_FUNCTION
                identifier = identifier.lower()
            elif identifier.lower() in SPECIAL_FUNCTIONS:
                id = TOKEN_ID.SPECIAL_FUNCTION
                identifier = identifier.lower()
            elif identifier.islower() and identifier.isalpha():
                id = TOKEN_ID.LOWERCASE_ALPHA
            else:
                id = TOKEN_ID.ALPHANUM
        # EMPTY
        elif self.current == None:
            id = TOKEN_ID.NONE
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
    example1 = 'declare a 1 b 1 expression a + b derivative wrt a'
    s = Scanner(example1)
    desc, ident = s.get_sym()
    while ident:
        print(f'{desc} {ident}')
        desc, ident = s.get_sym()
