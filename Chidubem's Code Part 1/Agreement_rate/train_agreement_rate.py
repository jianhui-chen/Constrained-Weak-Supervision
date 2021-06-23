import numpy as np
from itertools import chain, combinations
from linprog_wrapper import *
import matplotlib.pyplot as plt


def objective_function(c, error_rate, agreement_rate, rho, lambd, gamma, a_eq, a_ub):
	"""
	:param c: length n vector of c values
	:type c: array
	:param error_rate: length n vector of error rates
	:type error_rate: array
	:param agreement_rate: length n vector of agreemnent rates between functions
	:type agreement_rate: array
	:param rho: Scalar tuning hyperparameter
	:param type: float
	:param lambd: length n vector of lagrangian equality penalty parameters
	:type lambd: array
	:param gamma: length n vector of lagrangian inequality penalty parameters
	:type gamma: array
	:return: Scalar value of objective function
	:rtype: float
	"""

	objective = np.dot(c.T, error_rate)
	eq_constraint = np.dot(a_eq, error_rate) - agreement_rate + 1
	lambd_term = np.dot(lambd.T, eq_constraint)
	eq_augmented_term = (rho/2) * eq_constraint.T.dot(eq_constraint)

	ineq_constraint = np.dot(a_ub, error_rate)
	#ineq_constraint = np.clip(ineq_constraint, 0, None)
	ineq_augmented_term = (rho/2) * ineq_constraint.T.dot(ineq_constraint)
	gamma_term = np.dot(gamma.T, ineq_constraint)
	return objective + lambd_term + eq_augmented_term + gamma_term + ineq_augmented_term


def min_agreement(error_rate, agreement_rate, n_functions, weak_signals):                                  
	"""
    :param error_rate: length n vector of error rates
    :type error_rate: array
    :param agreement_rate: length n vector of agreemnent rates between functions
    :type agreement_rate: array
    :param n_functions: number of functions
    :type n_functions: integer
    :return: vector containing minimised agreement rates
    :rtype: array
    """

	def norm(vector, dim=None):
		"""
		:param vector: arbitrally vector of values
		:type vector: array
		:return: l2 norm of vector
		:rtype: float
		"""
		return np.sqrt(np.sum(vector**2))

	def check_tolerance(vec_t, vec_t2):
		"""
		:param vec_t, vec_t: vectors at different timesteps
		:type vector: array
		:return: boolean value, True if the vectors are equal within a tolerance level
		:rtype: boolean
		"""
		tol = 1e-8
		diff  = norm(vec_t - vec_t2)
		return np.abs(diff) < tol

	a_eq, b = Ab_eqmatrix(n_functions, agreement_rate)
	a_ub = Aub_matrix(n_functions)

	lambd = np.zeros(b.size)
	gamma = np.zeros(a_ub.shape[0])
	c, weak_array = initialize_c(weak_signals, n_functions)
	index = np.where(weak_array > 0)

	#aa_1 = np.linalg.pinv(a_eq.T.dot(a_eq))

	lagrangian = []

	max_iter = 100000
	t = 0
	converged = False
	while not converged and t < max_iter:
		rho = 0.9
		rate = 1 / np.sqrt(1 + t)
		# update error rate
		b = agreement_rate - 1
		old_error = error_rate

		#augmented term for equality constraint
		lambd_grad = a_eq.dot(error_rate) - b
		eq_augmented_term = a_eq.T.dot(lambd) + rho * a_eq.T.dot(lambd_grad)
		
		#augmented term for inequality constraint
		ue = np.dot(a_ub, error_rate)
		ineq_augmented_term = (a_ub.T).dot(gamma) + (a_ub.T).dot(ue.clip(0, None))

		# gradient step update
		e_grad = c + eq_augmented_term + ineq_augmented_term
		error_rate = error_rate - rate * e_grad

		# clip new_error to [0, 1] + weak signals to satisfy constraint
		error_rate[index] = error_rate[index].clip(max=weak_signals, min=0)
		error_rate = np.clip(error_rate, 0, 1)

		# update Lagrange multipliers
		old_lambd = lambd
		lambd = lambd + rate * lambd_grad

		old_gamma = gamma 
		gamma = gamma + rate * ue
		gamma = gamma.clip(0, None)

		# update agreement rate
		a_grad = -1 * (old_lambd + rho * lambd_grad) 
		old_agreement = agreement_rate
		# agreement_rate = agreement_rate + rate * a_grad
		# clip new_agreement to [0, 1] to satisfy constraint
		agreement_rate = np.clip(agreement_rate, 0, 1)

		conv_lambd = norm(np.dot(a_eq, error_rate) - b)
		conv_gamma = norm(np.dot(a_ub, error_rate).clip(min=0))

		converged = check_tolerance(error_rate, old_error) and np.isclose(conv_lambd, 0) and \
						np.isclose(conv_gamma, 0) and check_tolerance(agreement_rate, old_agreement)

		lagrangian += [objective_function(c, error_rate, agreement_rate, rho, lambd, gamma, a_eq, a_ub)]

		# print("Iter %d. Equality infeas: %f, Ineq infeas: %f" % (t, conv_lambd, conv_gamma))
	
		t += 1

	# hack to plot objective around current error_rate
	# vector = np.random.randn(error_rate.size)
	# eps = np.linspace(-1, 1, 100)
	# obj_val = np.zeros(eps.size)
	#
	# for i in range(eps.size):
	# 	obj_val[i] = objective_function(c, error_rate + eps[i] * vector, agreement_rate, rho, lambd, gamma, a_eq, a_ub)
	#
	# plt.plot(eps, obj_val)
	# plt.show()

	# print objective
	plt.plot(lagrangian)
	plt.show()

	eq_cons = np.dot(a_eq, error_rate)
	ineq_cons = np.dot(a_ub, error_rate)


	print('Estimated error rate:', error_rate)
	# print("Equality infeasibility:", eq_cons, b)
	# print("Inequality infeasibily:" + repr(ineq_cons))

	assert np.allclose(eq_cons,b)
	ineq_cons[ineq_cons < 0] = 0
	assert np.allclose(ineq_cons, 0)
	print('Minimized agreement rate:')
	return agreement_rate


def initialize_c(w_signal_bound, n_functions):
	"""
    :param w_signal_bound: list containing the upper bound error rate values of the weak signals 
    :type w_signal_bound: list
    :param n_functions: number of functions
    :type n_functions: integer
    :return: tuple of vector of c and weak signals
    :rtype: array, array
    """

	n_set = 2**n_functions - 1
	number_of_wfunctions = len(w_signal_bound)

	c = np.zeros(n_set)
	ub = np.zeros(n_set)
	n_learnable = n_functions - number_of_wfunctions
	for k in range(n_learnable):
		c[k] = -1

	j = 0
	for i in range(n_learnable, n_functions):
		ub[i] = w_signal_bound[j]
		j += 1

	return c, ub