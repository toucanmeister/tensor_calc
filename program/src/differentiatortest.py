import unittest
from differentiator import Differentiator

class DifferentiatorTests(unittest.TestCase):
    def test_base(self):
        test = 'declare a 0 argument a expression a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '_delta(0)')

    def test_product_1(self):
        test = 'declare a 1 b 1 c 0 argument c expression (a*(i,i->)b)*(,->)c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) *(,->) (a *(i,i->) b))')
    
    def test_product_2(self):
        test = 'declare a 0 argument a expression (a*(,->)a)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(0) *(,->) a) + (_delta(0) *(,->) a))')
    
    def test_product_3(self):
        test = 'declare a 0 b 0  argument a expression (b*(,->)a) *(,->) a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((_delta(0) *(,->) a) *(,->) b) + (_delta(0) *(,->) (b *(,->) a)))')

    def test_sum_1(self):
        test = 'declare a 0 b 0  argument a expression a + b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '_delta(0)')
    
    def test_sum_2(self):
        test = 'declare a 0 b 0  argument a expression a + a + b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + _delta(0))')
    
    def test_sum_3(self):
        test = 'declare a 0 b 0  argument a expression a + b + a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + _delta(0))')
    
    def test_difference_1(self):
        test = 'declare a 0 b 0  argument b expression a - b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) *(,->) (-(_IDENTITY_1)))')
    
    def test_difference_2(self):
        test = 'declare a 0 b 0  argument a expression a - a - b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + (_delta(0) *(,->) (-(_IDENTITY_1))))')
    
    def test_difference_3(self):
        test = 'declare a 0 b 0  argument a expression a - b - a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + (_delta(0) *(,->) (-(_IDENTITY_1))))')
    
    def test_sum_product_1(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)a + b + c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,i->ai) a) + (_delta(2) *(ai,i->ai) a))')
    
    def test_sum__product_2(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)b + a*(i,i->i)c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,i->ai) b) + (_delta(2) *(ai,i->ai) c))')
    
    def test_sum_product_3(self):
        test = 'declare a 1 b 1 c 0 argument a expression a*(i,i->)b + c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) *(,i->i) b)')
    
    def test_sum_product_4(self):
        test = 'declare a 2 b 1 argument a expression a*(ij,j->i)b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ai,j->aij) b)')
    
    def test_sum_product_5(self):
        test = 'declare a 2 b 1 argument a expression a*(ij,j->i)b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ai,j->aij) b)')
    
    def test_sum_product_6(self):
        test = 'declare A 2 B 2 x 1 argument x expression A*(ij,j->i)x + B*(ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,ij->aj) A) + (_delta(2) *(ai,ij->aj) B))')
    
    def test_sum_product_7(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A+B)*(ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ai,ij->aj) (A + B))')
    
    def test_sum_product_8(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A*(ij,ij->ij)B) * (ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ai,ij->aj) (A *(ij,ij->ij) B))')
    
    def test_sum_product_9(self):
        test = 'declare x 1 argument x expression x*(i,i->)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(0) *(,i->i) x) + (_delta(0) *(,i->i) x))')
    
    def test_sum_product_10(self):
        test = 'declare a 1 b 1 c 1 d 1 argument a expression a*(i,i->)b + a*(i,i->)c + a*(i,i->)d'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((_delta(0) *(,i->i) b) + (_delta(0) *(,i->i) c)) + (_delta(0) *(,i->i) d))')

    def test_difference_product_1(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)a - b - c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,i->ai) a) + (_delta(2) *(ai,i->ai) a))')
    
    def test_difference__product_2(self):
        test = 'declare a 1 b 1 c 1 argument a expression a*(i,i->i)b - a*(i,i->i)c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,i->ai) b) + ((_delta(2) *(ba,a->ba) (-(_IDENTITY_1))) *(ai,i->ai) c))')

    def test_difference_product_3(self):
        test = 'declare A 2 B 2 x 1 argument x expression (A-B)*(ij,j->i)x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ai,ij->aj) (A + (-(B))))')
    
    def test_sin_1(self):
        test = 'declare x 1 argument x expression sin(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (cos(x)))')

    def test_sin_2(self):
        test = 'declare A 2 x 1 argument x expression A *(ij,j->i) (sin(x))'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,ij->aj) A) *(ba,a->ba) (cos(x)))')
    
    def test_cos(self):
        test = 'declare A 2 x 1 argument x expression sin( A*(ij,j->i)x )'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ba,a->ba) (cos((A *(ij,j->i) x)))) *(ai,ij->aj) A)')
    
    def test_sin_cos(self):
        test = 'declare x 1 argument x expression cos(x) + sin(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ba,a->ba) (-((sin(x))))) + (_delta(2) *(ba,a->ba) (cos(x))))')
    
    def test_exp_1(self):
        test = 'declare x 1 argument x expression exp(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (exp(x)))')

    def test_exp_2(self):
        test = 'declare A 2 x 1 argument x expression A *(ij,j->i) exp(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,ij->aj) A) *(ba,a->ba) (exp(x)))')

if __name__ == '__main__':
    unittest.main()
