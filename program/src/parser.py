from scanner import *
from tree import *

# Important convention: Unary functions have their argument in 'right' and None in 'left'

class Parser():
    def __init__(self, input):
        self.dag = None    # Stores expression as a binary tree (Until CSE, then it's a binary DAG)
        self.variable_ranks = {}   # Stores declared variables and their ranks
        self.arg_name = None   # Stores the derivation argument variable name
        self.arg = None   # Stores the derivation argument variable
        self.scanner = Scanner(input)
        self.get_sym()
    
    def get_sym(self):
        (self.desc, self.ident) = self.scanner.get_sym()

    def fits(self, symbol):
        return self.desc == symbol
    
    def error(self, expected):
        raise Exception(f'Expected {expected} but found \'{self.ident}\'')

    def parse(self, clean=True):
        self.declaration()
        self.dag = self.expressionpart()
        self.argument()
        if clean:
            self.dag.eliminate_common_subtrees()
        self.dag.add_incoming_edges()
        self.dag.set_tensorrank(self.variable_ranks, self.arg)
        self.arg = self.dag.find(self.arg_name)
        self.split_double_powers()
        self.split_adj()
        self.dag = self.dag.fix_missing_indices(self.arg)
        self.dag.add_incoming_edges()
        self.dag.set_tensorrank(self.variable_ranks, self.arg)
        self.dag.unify_axes()
        self.arg = self.dag.find(self.arg_name) # Call this again since arg-subtree may have changed

    def declaration(self):
        if self.fits(TOKEN_ID.DECLARE):
            self.get_sym()
            self.tensordeclaration()
            while not self.fits(TOKEN_ID.EXPRESSION):
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
        if self.fits(TOKEN_ID.DERIVATIVE):
            self.get_sym()
            if self.fits(TOKEN_ID.WRT):
                self.get_sym()
                if self.fits(TOKEN_ID.ALPHANUM) or self.fits(TOKEN_ID.LOWERCASE_ALPHA):
                    self.arg_name = self.ident
                else:
                    self.error(TOKEN_ID.ALPHANUM.value)
            else:
                self.error(TOKEN_ID.WRT.value)
        else:
            self.error(TOKEN_ID.ARGUMENT.value)
        self.get_sym()
        if not self.desc == TOKEN_ID.NONE:
            raise Exception('Expected one argument to differentiate with respect to, but found multiple.')
    
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
                tree = Tree(NODETYPE.PRODUCT, '_TO_BE_SET_ELEMENTWISE', tree, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'elementwise_inverse', None, self.factor())) # Indices will get set in set_tensorrank
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
                tree = Tree(NODETYPE.POWER, '^', tree, self.expr())
                if self.fits(TOKEN_ID.RRBRACKET):
                    self.get_sym()
                else:
                    self.error(TOKEN_ID.RRBRACKET.value)
            else:
                tree = Tree(NODETYPE.POWER, '^', tree, self.atom())
        return tree
    
    def atom(self):
        if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
            tree = Tree(NODETYPE.CONSTANT, f'{self.ident}_{Tree.new_constant()}')
            self.get_sym()
        elif self.fits(TOKEN_ID.MINUS):
            self.get_sym()
            if self.fits(TOKEN_ID.CONSTANT) or self.fits(TOKEN_ID.NATNUM):
                tree = Tree(NODETYPE.ELEMENTWISE_FUNCTION, '-', None, Tree(NODETYPE.CONSTANT, f'{self.ident}_{Tree.new_constant()}'))
                self.get_sym()
            else:
                self.error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.NATNUM.value)
        elif self.fits(TOKEN_ID.ELEMENTWISE_FUNCTION):
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
        elif self.fits(TOKEN_ID.SPECIAL_FUNCTION):
            functionName = self.ident
            self.get_sym()
            if self.fits(TOKEN_ID.LRBRACKET):
                self.get_sym()
                tree = Tree(NODETYPE.SPECIAL_FUNCTION, functionName, None, self.expr())
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
            self.error(TOKEN_ID.CONSTANT.value + ' or ' + TOKEN_ID.ELEMENTWISE_FUNCTION.value + ' or ' + 'tensorname' +  ' or ' + TOKEN_ID.LRBRACKET.value)
        return tree

    def new_constant(self):
        constant = self.constant_id_counter
        self.constant_id_counter += 1
        return constant

    def split_double_powers(self):
        def create_split_power(node):
            indices = ''.join([i for i in string.ascii_lowercase][0:node.left.rank])
            prod = Tree(NODETYPE.PRODUCT, f'*(,{indices}->{indices})', node.right, Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'log', None, node.left))
            prod.set_indices('', indices, indices)
            return Tree(NODETYPE.ELEMENTWISE_FUNCTION, 'exp', None, prod)

        if self.dag.type == NODETYPE.POWER and self.dag.left.contains(self.arg) and self.dag.right.contains(self.arg):
            self.dag = create_split_power(self.dag)

        def split_powers_helper(node):
            if node.left: split_powers_helper(node.left)
            if node.right: split_powers_helper(node.right)
            if node.left and node.left.type == NODETYPE.POWER and node.left.left.contains(self.arg) and node.left.right.contains(self.arg):
                node.left = create_split_power(node.left)
            if node.right and node.right.type == NODETYPE.POWER and node.right.left.contains(self.arg) and node.right.right.contains(self.arg):
                node.right = create_split_power(node.right)
        split_powers_helper(self.dag)
    
    def split_adj(self):
        if self.dag.type == NODETYPE.SPECIAL_FUNCTION and self.dag.name == 'adj':
            self.dag = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, self.dag.right), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, self.dag.right))
            self.dag.set_indices('', 'ij', 'ij')
        def split_adj_helper(node):
            if node.left: split_adj_helper(node.left)
            if node.right: split_adj_helper(node.right)
            if node.left and node.left.type == NODETYPE.SPECIAL_FUNCTION and node.left.name == 'adj':
                node.left = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, node.left.right), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, node.left.right))
                node.left.set_indices('', 'ij', 'ij')
            if node.right and node.right.type == NODETYPE.SPECIAL_FUNCTION and node.right.name == 'adj':
                node.right = Tree(NODETYPE.PRODUCT, '*(,ij->ij)', Tree(NODETYPE.SPECIAL_FUNCTION, 'det', None, node.right.right), Tree(NODETYPE.SPECIAL_FUNCTION, 'inv', None, node.right.right))
                node.right.set_indices('', 'ij', 'ij')
        split_adj_helper(self.dag)

if __name__ == '__main__':
    example = 'declare a 0 b 0 expression a + b + a derivative wrt a'
    p = Parser(example)
    p.parse()
    p.dag.dot('dags/dag')