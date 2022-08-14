import unittest
from differentiator import Differentiator
from tree import NODETYPE, Tree
from numcheck import numcheck

class DifferentiatorTests(unittest.TestCase):
    numcheck_h = 1e-8
    numcheck_err_limit = 1e-6

    def reset_tree_attributes(self):
        Tree.reset_tree_attributes()

    def test_base(self):
        self.reset_tree_attributes()
        test = 'declare a 0 expression a derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'delta(0)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_product_1(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 0 expression (a*(i,i->)b)*(,->)c derivative wrt c'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(a *(i,i->) b)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_product_2(self):
        self.reset_tree_attributes()
        test = 'declare a 0 expression (a*(,->)a) derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(a + a)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_product_3(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression (b*(,->)a) *(,->) a derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((a *(,->) b) + (b *(,->) a))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))


    def test_sum_1(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a + b derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'delta(0)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_2(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a + a + b derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(0) + delta(0))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_3(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a + b + a derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(0) + delta(0))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_difference_1(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a - b derivative wrt b'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '-1')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_difference_2(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a - a - b derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(0) + -1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_difference_3(self):
        self.reset_tree_attributes()
        test = 'declare a 0 b 0 expression a - b - a derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(0) + -1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_1(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 1 expression a*(i,i->i)a + b + c derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,i->ai) a) + (delta(1) *(ai,i->ai) a))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum__product_2(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 1 expression a*(i,i->i)b + a*(i,i->i)c derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,i->ai) b) + (delta(1) *(ai,i->ai) c))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_3(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 0 expression a*(i,i->)b + c derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), 'b')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_4(self):
        self.reset_tree_attributes()
        test = 'declare a 2 b 1 expression a*(ij,j->i)b derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,j->aij) b)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_5(self):
        self.reset_tree_attributes()
        test = 'declare a 2 b 1 expression a*(ij,j->i)b derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,j->aij) b)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_6(self):
        self.reset_tree_attributes()
        test = 'declare A 2 B 2 x 1 expression A*(ij,j->i)x + B*(ij,j->i)x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,ij->aj) A) + (delta(1) *(ai,ij->aj) B))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_7(self):
        self.reset_tree_attributes()
        test = 'declare A 2 B 2 x 1 expression (A+B)*(ij,j->i)x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,ij->aj) (A + B))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_8(self):
        self.reset_tree_attributes()
        test = 'declare A 2 B 2 x 1 expression (A*(ij,ij->ij)B) * (ij,j->i)x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,ij->aj) (A *(ij,ij->ij) B))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_9(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression x*(i,i->)x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(x + x)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sum_product_10(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 1 d 1 expression a*(i,i->)b + a*(i,i->)c + a*(i,i->)d derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((b + c) + d)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_difference_product_1(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 1 expression a*(i,i->i)a - b - c derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,i->ai) a) + (delta(1) *(ai,i->ai) a))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_difference__product_2(self):
        self.reset_tree_attributes()
        test = 'declare a 1 b 1 c 1 expression a*(i,i->i)b - a*(i,i->i)c derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,i->ai) b) + ((delta(1) *(ba,a->ba) -1) *(ai,i->ai) c))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_difference_product_3(self):
        self.reset_tree_attributes()
        test = 'declare A 2 B 2 x 1 expression (A-B)*(ij,j->i)x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,ij->aj) (A - B))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sin_1(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression sin(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (cos(x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_sin_2(self):
        self.reset_tree_attributes()
        test = 'declare A 2 x 1 expression A *(ij,j->i) (sin(x)) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,ij->aj) A) *(ba,a->ba) (cos(x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sin_3(self):
        self.reset_tree_attributes()
        test = 'declare A 2 x 1 expression sin( A*(ij,j->i)x ) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ba,a->ba) (cos((A *(ij,j->i) x)))) *(ai,ij->aj) A)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sin_cos(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression cos(x) + sin(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ba,a->ba) (-((sin(x))))) + (delta(1) *(ba,a->ba) (cos(x))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_exp_1(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression exp(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (exp(x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_exp_2(self):
        self.reset_tree_attributes()
        test = 'declare A 2 x 1 expression A *(ij,j->i) exp(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ai,ij->aj) A) *(ba,a->ba) (exp(x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_div(self):
        self.reset_tree_attributes()
        test = 'declare y 1 x 1 expression y / x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(1) *(ba,a->ba) y) *(ba,a->ba) (-((1 / ((x *(a,a->a) x))))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_tan(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression tan(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (1 / (((cos(x)) *(a,a->a) (cos(x))))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_arcsin(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression arcsin(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (1 / (((1 - (x ^ 2)) ^ 0.5))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_arccos(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression arccos(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (-((1 / (((1 - (x ^ 2)) ^ 0.5))))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_arctan(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression arctan(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (1 / (((x *(a,a->a) x) + 1))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_log(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression log(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (1 / (x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_tanh(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression tanh(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (1 - ((tanh(x)) *(a,a->a) (tanh(x)))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_abs(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression abs(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (sign(x)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_sign(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression sign(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) 0)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_relu(self):
        self.reset_tree_attributes()
        test = 'declare x 1 expression relu(x) derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (relu((sign(x)))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_power_1(self):
        self.reset_tree_attributes()
        test = 'declare x 1 a 0 expression x^a derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ba,a->ba) (a *(,a->a) (x ^ (a - 1))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_power_2(self):
        self.reset_tree_attributes()
        test = 'declare x 1 a 0 expression (x^a) *(i,i->) x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((x *(a,a->a) (a *(,a->a) (x ^ (a - 1)))) + (x ^ a))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_power_3(self):
        self.reset_tree_attributes()
        test = 'declare x 0 a 1 expression a^x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((a ^ x) *(b,b->b) (log(a)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_power_4(self):
        self.reset_tree_attributes()
        test = 'declare x 0 a 1 expression x^x derivative wrt x'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((x ^ x) *(,->) (log(x))) + (((x ^ x) *(,->) x) *(,->) (1 / (x))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_inv(self):
        self.reset_tree_attributes()
        test = 'declare A 2 expression inv(A) derivative wrt A'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(2) *(efcd,cdab->efab) ((-((inv(A)))) *(ij,kl->kjli) (inv(A))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_det(self):
        self.reset_tree_attributes()
        test = 'declare A 2 expression det(A) derivative wrt A'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((adj(A)) *(ij,->ji) 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_adj(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression adj(X) derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((inv(X)) *(cd,ab->cdab) ((adj(X)) *(ij,->ji) 1)) + ((delta(2) *(abij,->abij) (det(X))) *(efcd,cdab->efab) ((-((inv(X)))) *(ij,kl->kjli) (inv(X)))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_missing_indices_1(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression 1*(,ij->)X derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(1 *(ij,->ij) 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_missing_indices_3(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression X + X derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(2) + delta(2))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_missing_indices_4(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression 1*(ij, ij -> )(X + X) derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(1 + 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_missing_indices_5(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression 1*(ij, ij -> )X + 1*(ij, ij -> )X derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(1 + 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_missing_indices_6(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression 1*(ij, ij -> )X + 1*(ij, ij -> )(X+1) derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(1 + 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_missing_indices_7(self):
        self.reset_tree_attributes()
        test = 'declare v 1 expression v*(i, i -> )v + 1*(i, i -> )(v) derivative wrt v'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((v + v) + 1)')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_adding_contribtutions(self):
        self.reset_tree_attributes()
        test = 'declare v 1 expression (v *(i,i->i) v) + (v *(i,i->i) v) derivative wrt v'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(((delta(1) + delta(1)) *(ai,i->ai) v) + ((delta(1) + delta(1)) *(ai,i->ai) v))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))
    
    def test_delta_1(self):
        self.reset_tree_attributes()
        test = 'declare v 1 expression (delta(0) + 1) *(,i->i) v derivative wrt v'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(delta(1) *(ai,->ai) (delta(0) + 1))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_delta_2(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression sin(delta(1)) *(ij,ij->) X derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '(sin(delta(1)))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_higher_dims(self):
        self.reset_tree_attributes()
        test = 'declare a 2 X 4 expression a*(ij,ijkl->ijkl)X + cos(X) derivative wrt X'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '((delta(4) *(abcdijkl,ij->abcdijkl) a) + (delta(4) *(efghabcd,abcd->efghabcd) (-((sin(X))))))')
        self.assertTrue(numcheck(d, h=self.numcheck_h, err_limit=self.numcheck_err_limit))

    def test_arg_missing(self):
        self.reset_tree_attributes()
        test = 'declare X 2 expression sin(delta(1)) *(ij,ij->) X derivative wrt a'
        d = Differentiator(test)
        d.differentiate()
        self.assertEqual(str(d.diffDag), '0')

if __name__ == '__main__':
    unittest.main()
