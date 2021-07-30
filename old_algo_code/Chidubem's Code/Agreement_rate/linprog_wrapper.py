"""Linear programming solution using linprog"""
from scipy.optimize import linprog
from numpy.linalg import solve
import numpy as np
from itertools import chain, combinations
import itertools


def solve_linprog(agreement_rate, w_signal_bound, n_functions, error_rate=None):
    """
    Linprog (https://docs.scipy.org/doc/scipy/reference/optimize.linprog-simplex.html) to make it more compatible with standard
    linear program constraint types. This function will solve
        minimize      c^T * x
        subject to  A_ub * x <= b_ub
        and         A_eq * x == b_eq
    
    Note: if the setup is incorrect, this method will catch a failure exception thrown by linprog,
    print a warning to the console, and return the all-zeros solution.
    :param agreement_rate: vector containing the agreement rates of combinations of functions
    :type agreement_rate: array
    :param w_signal_bound: list containing the upper bound error rate values of the weak signals 
    :type w_signal_bound: list
    :param n_functions: number of functions
    :type n_functions: integer
    :param error_rate: vector containing the true error rates of the functions
    :type error_rate: array
    :return: estimated error rates of the functions
    :rtype: array
    """
    n_set = 2**n_functions - 1
    
    A_ub = Aub_matrix(n_functions)
    size,_ = A_ub.shape
    b_ub = np.zeros(size)
    A_eq, b_eq = Ab_eqmatrix(n_functions, agreement_rate)
    c, ub = initilize_weak_signal(w_signal_bound, n_functions)
    lb = np.zeros(n_set)
    
    """
    if error_rate is not None:
        print("Error rate was provided. Checking feasibility")
        print("Equality constraint:")
        print(A_eq.dot(error_rate), b_eq)
        print("Inequality constraint: ")
        print(A_ub.dot(error_rate), b_ub)
	"""
	
    # solve linear program
    try:
        sol = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)))
    except ValueError:
        print("Warning: Linear program solver exited with a numerical error "
              "(Linprog can be sensitive to mistakes). Guessing all-zeros solution.")
        sol = np.zeros(n_set)

    return sol


def all_subsets(s):
    """
    :param s: vector of individual functions
    :type s: array
    :return: a generator object containing all subsets of s
    :rtype: object
    """
    return chain(*map(lambda x: combinations(s, x), range(1, len(s)+1)))


def Aub_matrix(n_functions):
    """
    :param n_functions: number of functions
    :type n_functions: integer
    :return: tuple of A_eq matrix and b_eq array
    :rtype: 2D array
    """

    function_array = np.arange(1, n_functions+1)
    au_matrix = []
    subset_index = {}
    n_set = 2**n_functions - 1
    for index, subset in enumerate(all_subsets(function_array)):
        subset_index[subset] = index
    
    for subs in all_subsets(function_array):
        if(len(subs)) > 1:
            setlist = list(subs)
            for i in range(len(subs)):                
                del setlist[i]
                temp = np.zeros(n_set)
                index = subset_index[subs]
                temp[index] = 1
                index = subset_index[tuple(setlist)]
                temp[index] = -1
                au_matrix.append(temp)
                setlist = list(subs)

    return np.array(au_matrix)


def Ab_eqmatrix(n_functions, agreement_rate):
    """
    :param n_functions: number of functions
    :type n_functions: integer
    :param agreement_rate: vector containing the agreement rates of combinations of functions
    :type agreement_rate: array
    :return: tuple of A_eq matrix and b_eq array
    :rtype: 2D array, array
    """

    function_array = np.arange(1, n_functions+1)
    col_size = 2**n_functions - 1
    b_size = (2**n_functions) - n_functions - 1
    ae_matrix = np.zeros((b_size,col_size))
    b_array = np.zeros(b_size)
    subset_index = {}
    for index, subset in enumerate(all_subsets(function_array)):
        subset_index[subset] = index
    
    count = 0
    for subs in all_subsets(function_array):
        if(len(subs)) > 1:
            setlist = list(subs)
            row = np.zeros(col_size)
            b_array[count] = agreement_rate[count] - 1
            index = subset_index[subs]
            row[index] = 1
            for k in range(1, len(subs)+1):
                val = (-1)**k
                for smaller_sub in itertools.combinations(subs, k):
                    index = subset_index[smaller_sub]
                    row[index] += val

            ae_matrix[count, :] = row
            count += 1

    return ae_matrix, b_array


def initilize_weak_signal(w_signal_bound, n_functions):
	"""
    :param w_signal_bound: vector containing the upper bound error rate values of the weak signals 
    :type w_signal_bound: array
    :param n_functions: number of functions
    :type n_functions: integer
    :return: tuple of c and upper bound vectors
    :rtype: array, array
    """

    
	n_set = 2**n_functions - 1
	number_of_wfunctions = len(w_signal_bound)

	c = np.zeros(n_set)
	ub = np.ones(n_set)
	n = n_functions - number_of_wfunctions
	for k in range(n):
	    c[k] = -1

	if number_of_wfunctions != 0:
		j = 0
		for i in range(n, n_functions):
			ub[i] = w_signal_bound[j]
			j += 1

	return c, ub

