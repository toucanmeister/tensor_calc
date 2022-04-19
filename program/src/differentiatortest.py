import unittest
from differentiator import Differentiator

class DifferentiatorTests(unittest.TestCase):
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

    def test_sum_1(self):
        test = 'declare a 0 b 0  argument a expression a + b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '_IDENTITY')
    
    def test_sum_2(self):
        test = 'declare a 0 b 0  argument a expression a + a + b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_IDENTITY + _IDENTITY)')
    
    def test_sum_3(self):
        test = 'declare a 0 b 0  argument a expression a + b + a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_IDENTITY + _IDENTITY)')
    
    def test_sum_product_1(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)a + b + c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(a + a)')
    
    def test_sum__product_2(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)b + a*(i,i->i)c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(c + b)')
    
    def test_sum_product_3(self):
        test = 'declare a 1 b 1 c 0 argument a expression a*(i,i->)b + c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'b')
    
    def test_sum_product_4(self):
        test = 'declare a 2 b 1 argument a expression a*(ij,j->i)b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'b')
    
    def test_sum_product_5(self):
        test = 'declare a 2 b 1 argument a expression a*(ij,j->i)b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'b')
    
    def test_sum_product_6(self):
        test = 'declare A 2 B 2 x 1 argument x expression A*(ij,j->i)x + B*(ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(B + A)')
    
    def test_sum_product_7(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A+B)*(ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(A + B)')
    
    def test_sum_product_8(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A*(ij,ij->ij)B) * (ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(A *(ij,ij->ij) B)')
    
    def test_sum_product_9(self):
        test = 'declare x 1 argument x expression x*(i,i->)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(x + x)')
    
    def test_sum_product_10(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A*(ij,j->i)x) *(i,i->) ((A+B)*(ij,j->i)x)'
        d = Differentiator(test)
        d.differentiate()
        print(d.diffDag)
        self.assertEqual(str(d.diffDag), '(((A *(ij,j->i) x) *(i,ij->j) (A + B)) + (((A + B) *(ij,j->i) x) *(i,ij->j) A))')


if __name__ == '__main__':
    unittest.main()
