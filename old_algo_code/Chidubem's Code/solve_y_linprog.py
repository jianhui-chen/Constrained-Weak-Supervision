"""Linear programming solution using linprog"""
from scipy.optimize import linprog
import numpy as np


def solve_linprog(learnable_probabilities, weak_signal_probabilities, weak_signal_bound):
    """
    Linprog (https://docs.scipy.org/doc/scipy/reference/optimize.linprog-simplex.html) to make it more compatible with standard
    linear program constraint types. This function will solve
        minimize      c^T * x
        subject to  A_ub * x <= b_ub
        and         A_eq * x == b_eq
    
    Note: if the setup is incorrect, this method will catch a failure exception thrown by linprog,
    print a warning to the console, and return the all-zeros solution.
    :param learnable_probabilities: vector containing the probabilities of the learnable classifier
    :type learnable_probabilities: array
    :param weak_signal_probabilities: ndarray of weak signal probabilities
    :type weak_signal_probabilities: ndarray
    :param weak_signal_bound: vector containing the upper bound error rate values of the weak signals 
    :type weak_signal_bound: array
    :return: worse case bounds for the labels y
    :rtype: array
    """
    c = 1 - 2 * learnable_probabilities
    n, d = weak_signal_probabilities.shape
    A_ub = 1 - 2 * weak_signal_probabilities
    b_ub = weak_signal_bound - weak_signal_probabilities.dot(np.ones(d))
    lb = np.zeros(d)
    ub = np.ones(d)
    #solve linear program
    sol = linprog(c, A_eq=None, b_eq=None, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)))
    """
    try:
        sol = linprog(c, A_eq=None, b_eq=None, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='interior-point')
    except ValueError:
        print("Warning: Linear program solver exited with a numerical error "
              "(Linprog can be sensitive to mistakes). Guessing all-zeros solution.")
        sol = np.zeros(n)
    """
    return sol


