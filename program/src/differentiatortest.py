import unittest
from differentiator import Differentiator

class ParserTests(unittest.TestCase):
    def test_base(self):
        test = 'declare a 0 argument a expression a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '_IDENTITY')

    def test_product_1(self):
        test = 'declare a 1 b 1 c 0 argument c expression (a*(i,i->)b)*(,->)c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(a *(i,i->) b)')
    
    def test_product_2(self):
        test = 'declare a 0 argument a expression (a*(,->)a)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(a + a)')
    
    def test_product_3(self):
        test = 'declare a 0 b 0  argument a expression (b*(,->)a) *(,->) a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((b *(,->) a) + (a *(,->) b))')


if __name__ == '__main__':
    unittest.main()
