import unittest
from parser import Parser

class ParserTests(unittest.TestCase):
    def test_base(self):
        test = 'declare a 0 argument a expression a'
        p = Parser(test)
        p.parse()
    def test_declarations(self):
        test = 'declare a0 0 b 10 tensor0dude 100 tensor1dude 1 argument tensor0dude expression a0'
        p = Parser(test)
        p.parse()
    def test_no_tensorrank(self):
        test1 = 'declare a0 argument a0 expression a0'
        test2 = 'declare a0 b1 1 argument b1 expression b1'
        self.assertRaises(Exception, Parser(test1).parse)
        self.assertRaises(Exception, Parser(test2).parse)
    def test_multiple_arguments(self):
        test = 'declare a 0 b 1 argument a b expression a'
        self.assertRaises(Exception, Parser(test).parse)
    def test_whitespace(self):
        test = '''
            declare
                a 1
                b 1
            argument a
            expression  a + b
                '''
        p = Parser(test)
        p.parse()

    def test_goodexpression_1(self):
        test = 'declare a 0 b 0 argument a expression 2 + a^(-1)+b'
        p = Parser(test)
        p.parse()
    def test_goodexpression_2(self):
        test = 'declare a 1 b 1 argument a expression a*(i,j->ij)b'
        p = Parser(test)
        p.parse()
    def test_scalar_product(self):
        test = 'declare a 0 b 0 argument a expression a*(,->)b'
        p = Parser(test)
        p.parse()
    def test_inner_product(self):
        test = 'declare a 1 b 1 argument a expression a*(i,i->)b'
        p = Parser(test)
        p.parse()
    def test_outer_product(self):
        test = 'declare a 1 b 1 argument a expression a*(i,j->ij)b'
        p = Parser(test)
        p.parse()
    def test_scalar_times_vector(self):
        test = 'declare a 0 b 1 argument a expression a*(,j->j)b'
        p = Parser(test)
        p.parse()
    def test_matrix_application(self):
        test = 'declare a 2 b 1 argument a expression a*(ij,j->i)b'
        p = Parser(test)
        p.parse()
    def test_matrix_multiplication(self):
        test = 'declare a 2 b 2 argument a expression a*(ik,kj->ij)b'
        p = Parser(test)
        p.parse()
    def test_elementwise_multiplication(self):
        test = 'declare a 3 b 3 argument a expression a*(ijk,ijk->ijk)b'
        p = Parser(test)
        p.parse()
    def test_badexpression_product(self):
        test = 'declare a 1 b 1 argument a expression a*(i,j->3)b'
        self.assertRaises(Exception, Parser(test).parse)
    def test_badexpression_function(self):
        test = 'declare a 0 b 0 argument a expression a-log()+b'
        self.assertRaises(Exception, Parser(test).parse)
    def test_badexpression_brackets(self):
        test = 'declare a 0 b 0 argument a expression (a+)b)'
        self.assertRaises(Exception, Parser(test).parse)
    

if __name__ == '__main__':
    unittest.main()