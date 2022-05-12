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
        self.assertEqual(str(d.diffDag), '(_delta(0) *(,->) (-(_ones(0))))')
    
    def test_difference_2(self):
        test = 'declare a 0 b 0  argument a expression a - a - b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + (_delta(0) *(,->) (-(_ones(0)))))')
    
    def test_difference_3(self):
        test = 'declare a 0 b 0  argument a expression a - b - a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(0) + (_delta(0) *(,->) (-(_ones(0)))))')
    
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
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ai,i->ai) b) + ((_delta(2) *(ba,a->ba) (-(_ones(1)))) *(ai,i->ai) c))')

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

    def test_div(self):
        test = 'declare y 1 x 1 argument x expression y / x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((_delta(2) *(ba,a->ba) y) *(ba,a->ba) (-((elementwise_inverse((x *(a,a->a) x))))))')

    def test_tan(self):
        test = 'declare x 1 argument x expression tan(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (elementwise_inverse(((cos(x)) *(a,a->a) (cos(x))))))')
    
    def test_arcsin(self):
        test = 'declare x 1 argument x expression arcsin(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (elementwise_inverse(((_ones(1) + (-((x ^ 2)))) ^ 0.5))))')
    
    def test_arccos(self):
        test = 'declare x 1 argument x expression arccos(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (-((elementwise_inverse(((_ones(1) + (-((x ^ 2)))) ^ 0.5))))))')

    def test_arctan(self):
        test = 'declare x 1 argument x expression arctan(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (elementwise_inverse(((x *(a,a->a) x) + _ones(1)))))')
    
    def test_log(self):
        test = 'declare x 1 argument x expression log(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (elementwise_inverse(x)))')

    def test_tanh(self):
        test = 'declare x 1 argument x expression tanh(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (_ones(1) + (-(((tanh(x)) *(a,a->a) (tanh(x)))))))')
    
    def test_abs(self):
        test = 'declare x 1 argument x expression abs(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (sign(x)))')
    
    def test_sign(self):
        test = 'declare x 1 argument x expression sign(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) _zeroes(1))')
    
    def test_relu(self):
        test = 'declare x 1 argument x expression relu(x)'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (relu((sign(x)))))')
    
    def test_power_1(self):
        test = 'declare x 1 a 0 argument x expression x^a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ba,a->ba) (a *(,a->a) (x ^ (a + (-(_ones(0)))))))')

    def test_power_2(self):
        test = 'declare x 1 a 0 argument x expression (x^a) *(i,i->) x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((_delta(0) *(,i->i) x) *(a,a->a) (a *(,a->a) (x ^ (a + (-(_ones(0))))))) + (_delta(0) *(,i->i) (x ^ a)))')

    def test_power_3(self):
        test = 'declare x 0 a 1 argument x expression a^x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(_delta(2) *(ab,b->a) ((a ^ x) *(b,b->b) (log(a))))')
    
    def test_power_4(self):
        test = 'declare x 0 a 1 argument x expression x^x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((_delta(0) *(,->) (exp((x *(,->) (log(x)))))) *(,->) (log(x))) + (((_delta(0) *(,->) (exp((x *(,->) (log(x)))))) *(,->) x) *(,->) (elementwise_inverse(x))))')

if __name__ == '__main__':
    unittest.main()
