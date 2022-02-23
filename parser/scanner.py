import string

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

ALPHA = list(string.ascii_letters)
DIGITS = [str(i) for i in range(10)]
SYMBOLS = {
    '+': 'plus',
    '-': 'minus',
    '*': 'multiply',
    '/': 'divide',
    '^': 'pow',
    '(': 'lrbracket',
    ')': 'rrbracket',
    '>': 'greater',
    ',': 'comma'
}
KEYWORDS = {'declare', 'argument', 'expression'}
FUNCTIONS = {'sin', 'cos', 'exp', 'log', 'norm2', 'tr', 'det', 'logdet', 'inv', 'sqrt', 'abs', 'diag'}

class Scanner():
    def __init__(self, input):
        self.input = Input(input)
        self.current = self.input.next()
    
    def get_sym(self):
        identifier = ''
        description = ''

        self.ignore_whitespace()
        
        # NUMBER
        if self.current in DIGITS:
            description = 'number'
            while self.current in DIGITS:
                identifier += self.current
                self.current = self.input.next()
        # SYMBOLS
        elif self.current in SYMBOLS.keys():
            description = SYMBOLS.get(self.current)
            identifier = self.current
            self.current = self.input.next()
        # KEYWORDS, FUNCTIONS, LOWERCASE_ALPHA AND OTHER WORDS
        elif self.current in ALPHA:
            while self.current in ALPHA or self.current in DIGITS:
                identifier += self.current
                self.current = self.input.next()
            if identifier in KEYWORDS:
                description = 'keyword'
            elif identifier in FUNCTIONS:
                description = 'function'
            elif identifier.islower() and identifier.isalpha():
                description = 'lowercase_alpha'
            elif identifier.isalpha():
                description = 'alpha'
            else:
                description = 'alphanum'
        elif self.current == None:
            description = 'none'
            identifier = None
        return description, identifier

    
    def ignore_whitespace(self):
        while self.current == ' ':
            self.current = self.input.next()

if __name__ == '__main__':
    example = 'declare a(ij) b(kl) c(mn) argument a expression a*(ij,jk->ik)b + c'
    s = Scanner(example)
    desc, ident = s.get_sym()
    while ident:
        print(f'{desc} {ident}')
        desc, ident = s.get_sym()
