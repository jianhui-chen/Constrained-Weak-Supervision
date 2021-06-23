"""Test class for principal components analysis and Gaussian mixture modeling."""
import unittest
import numpy as np
from itertools import chain, combinations
from linprog_wrapper import *
from train_agreement_rate import *


class Testlinprog(unittest.TestCase):
    """Tests for PCA and GMMs"""
    def setUp(self):
        """
        Create agreement rate vector and dictionary based on the number of classifiers
        :return: None
        """
        n = 4 # no of classifiers
        d = 20 # no of test examples
        p = .7 # prob of .7 meaning that the classifiers perform better than chance
        number_of_wsignals = 2
        weak_signals = []
        p_matrix = np.random.binomial(1, p, size=(n, d)) # matrix containing d binary predictions from n classifiers

        s = np.arange(n)
        subs = [(j) for i in range(1,len(s)) for j in combinations(s, i+1)]

        agreement_rate = {}
        ar = []

        error_rate = []

        for i in s:
            error_rate.append(1 - np.sum(p_matrix[i, :]) / d)

        diff = n - number_of_wsignals
        
        for k in range(diff, n):
            weak_signals.append(error_rate[k])

        for x in subs:
            a = x[0]
            b = x[1]
            temp = np.where(p_matrix[a,:] == p_matrix[b,:])[0]
            count = len(temp)
            if(len(x)) > 2:
                for k in range(2, len(x)):
                    b = x[k-1]
                    c = x[k]
                    var = np.where(p_matrix[b,:] == p_matrix[c,:])[0]
                    temp = [item for item in var if item in temp]
                count = len(temp)
            agreement_rate[x] = count / d
            ar.append(count / d)

            er = 1 - np.sum(np.max(p_matrix[x, :], axis=0)) / d
            error_rate.append(er)

        self.agreement_rate = agreement_rate
        self.ar = np.array(ar)
        self.size = n
        self.weak_signals = np.array(weak_signals)
        self.error_rate = np.array(error_rate)
        #print("Agreement rate:")
        #print(ar)
       

    def test_solve_linprog(self):
        """
        Solves the linprog optimization using generated agreement rate
        :return: None
        """
        print('weak signals error bound:', np.array(self.weak_signals))
        print("True error rate:", self.error_rate)
        print("Linprog solution")
        sol = solve_linprog(self.ar, self.weak_signals, self.size, self.error_rate)
        print("Estimated error rate:")
        print(sol.x)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
        """
        Tests the correctness of our implementation
        """

        n_functions = self.size
        n_set = 2**n_functions - 1
        weak_signals = self.weak_signals
        agreement_rate = self.ar
        error_rate = np.random.rand(n_set)
        print("Train agreement solution:")
        print(min_agreement(error_rate, agreement_rate, n_functions, weak_signals))
        print('True agreement rate:', self.ar)


    def test_all_subsets(self):
        """
        Tests that all subsets returns the right subsets.
        :return: None
        """
        n = 3
        sample = np.arange(1,n+1)
        subsets = []
        for subset in all_subsets(sample):
            subsets.append(subset)
        set_list = np.array([(1,),(2,),(3,),(1,2),(1,3),(2,3),(1,2,3)])
        assert set_list.all() == np.asarray(subsets).all(), "The subsets are not correctly created"        
     

if __name__ == '__main__':
    unittest.main()
