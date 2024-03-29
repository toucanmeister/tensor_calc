import unittest
from parser import parse

class ParserTests(unittest.TestCase):
    def test_base(self):
        test = 'declare a 0 expression a derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), 'a')
    def test_declarations(self):
        test = 'declare a0 0 b 10 tensor0test 100 tensor1test 1 expression a0 derivative wrt tensor0test'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), 'a0')
    def test_no_tensorrank(self):
        test1 = 'declare a0 expression a0 derivative wrt a0'
        test2 = 'declare a0 b1 1 expression b1 derivative wrt b1'
        self.assertRaises(Exception, parse, test1)
        self.assertRaises(Exception, parse, test2)
    def test_multiple_arguments(self):
        test = 'declare a 0 b 1 expression a derivative wrt a b'
        self.assertRaises(Exception, parse, test)
    def test_whitespace(self):
        test = '''
            declare
                a 1
                b 1
            expression  a + b
            derivative wrt a
                '''
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a + b)')
    def test_goodexpression_1(self):
        test = 'declare a 0 b 0 expression 2 + a^(-1)+b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '((2 + (a ^ (-(1)))) + b)')
    def test_goodexpression_2(self):
        test = 'declare a 1 b 1 expression a*(i,j->ij)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(i,j->ij) b)')
    def test_goodexpression_3(self):
        test = 'declare a 3 b 3 c 0 expression ((a / b) *(ijk,ijk->) a) + (-(c)) derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(((a _TO_BE_SET_ELEMENTWISE (1 / (b))) *(ijk,ijk->) a) + (-(c)))')
    def test_goodexpression_4(self):
        test = 'declare x 1 a 0 expression x^(x *(i,i->) x + a) derivative wrt x'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(x ^ ((x *(i,i->) x) + a))')
    def test_scalar_product(self):
        test = 'declare a 0 b 0 expression a*(,->)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(,->) b)')
    def test_inner_product(self):
        test = 'declare a 1 b 1 expression a*(i,i->)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(i,i->) b)')
    def test_outer_product(self):
        test = 'declare a 1 b 1 expression a*(i,j->ij)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(i,j->ij) b)')
    def test_scalar_times_vector(self):
        test = 'declare a 0 b 1 expression a*(,j->j)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(,j->j) b)')
    def test_matrix_application(self):
        test = 'declare a 2 b 1 expression a*(ij,j->i)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(ij,j->i) b)')
    def test_matrix_multiplication(self):
        test = 'declare a 2 b 2 expression a*(ik,kj->ij)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(ik,kj->ij) b)')
    def test_elementwise_multiplication(self):
        test = 'declare a 3 b 3 expression a*(ijk,ijk->ijk)b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a *(ijk,ijk->ijk) b)')
    def test_difference(self):
        test = 'declare a 2 b 2 expression a - b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a + (-(b)))')
    def test_quotient(self):
        test = 'declare a 3 b 3 expression a / b derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a _TO_BE_SET_ELEMENTWISE (1 / (b)))')
    def test_special_function(self):
        test = 'declare a 2 expression inv(a) derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(inv(a))')
    def test_badexpression_product(self):
        test = 'declare a 1 b 1 expression a*(i,j->3)b derivative wrt a'
        self.assertRaises(Exception, parse, test)
    def test_badexpression_function(self):
        test = 'declare a 0 b 0 expression a-log()+b derivative wrt a'
        self.assertRaises(Exception, parse, test)
    def test_badexpression_brackets(self):
        test = 'declare a 0 b 0 expression (a+)b) derivative wrt a'
        self.assertRaises(Exception, parse, test)
    def test_broadcasting_1(self):
        test = 'declare a 2 expression a + 2 derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(a + 2)')
    def test_broadcasting_2(self):
        test = 'declare A 2 x 1 expression (A + 1) *(ij,j->i) 1  derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '((A + 1) *(ij,j->i) 1)')
    def test_delta_1(self):
        test = 'declare A 2 expression delta(0) *(,ij->) A  derivative wrt a'
        dag, _, _ = parse(test)
        self.assertEqual(str(dag), '(delta(0) *(,ij->) A)')
    
if __name__ == '__main__':
    unittest.main()