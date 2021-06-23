import numpy as np
from itertools import chain, combinations
from linprog_wrapper import *
import matplotlib.pyplot as plt


def objective_function(c, error_rate, agreement_rate, rho, lambd, gamma, a_eq, a_ub):
	"""
	Computes the value of the objective function

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


def logistic(x):
    """
    Squashing function that returns the squashed value and a vector of the
                derivatives of the squashing operation.

    :param x: ndarray of inputs to be squashed
    :type x: ndarray
    :return: tuple of (1) squashed inputs and (2) the gradients of the
                squashing function, both ndarrays of the same shape as x
    :rtype: tuple
    """
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)

    return y, grad


def probability(data, weights, weak_signal_probabilities):
	"""
	Computes the probabilities of the data instance 

	:param data: size (d, n) ndarray containing n examples described by d features each
	:type data: ndarray
	:param weights: size (n_learnable_functions, d) containing vectors of weights for each function
	:type weights: ndarray
	:param weak_signal_probabilities: list containing weak signal peobabilities
	:type weak_signal_probabilities: list 
	:return: ndarray of size (n_functions, n) prababilities for each data point of each function
	:rtype: ndarray
	"""

	probs = []
	for weight in weights:
		y = weight.dot(data)
		p, _ = logistic(y)
		probs.append(p)

	probs = np.vstack((probs, weak_signal_probabilities))
	return probs


def dp_dw_gradient(data, weights):
	"""
	Computes the gradient the probabilities wrt to the weights 

	:param data: size (d, n) ndarray containing n examples described by d features each
	:type data: ndarray
	:param weights: size (n_learnable_functions, d) containing vectors of weights for each learnable function
	:type weights: ndarray 
	:return: ndarray of size (n_of_features, n) gradients for each probability wrt to weight
	:rtype: ndarray
	"""

	w_gradient = []
	for weight in weights:
		y = weight.dot(data)
		_, grad = logistic(y)
		grad = data * grad
		w_gradient.append(grad)

	return np.array(w_gradient)
		

def calculate_agreement_rate(probs):
	"""
	Computes the values of agreement rate vector 

	:param probs: size (n_functions, n) prababilities for each data point of each function
	:type probs: ndarray
	:return: vector of agreement rates
	:rtype: array
	"""
	n_classifiers = probs.shape[0]
	agreement_rate = []

	s = np.arange(n_classifiers)
	classifier_inds = [(j) for i in range(1,len(s)) for j in combinations(s, i+1)]

	for item in classifier_inds:
		temp = np.prod(probs[item, :], axis=0)
		temp_cl = np.prod(1 - probs[item, :], axis=0)

		agreement_item = np.sum(temp + temp_cl) / temp.size
		agreement_rate.append(agreement_item)

	return np.array(agreement_rate)


def da_dp_gradient(probs, n_learnable):
	"""
	Computes the gradient of agreement rate wrt probabilities 

	:param probs: size (n_functions, n) prababilities for each data point of each function
	:type probs: ndarray
	:param n_learnable: the number of learnable functions
	:type n_learnable: int 
	:return: size (n_functions, agreement_rate.size) of probability gradients
	:rtype: ndarray
	"""
	n_classifiers, n = probs.shape
	gradient = []

	s = np.arange(n_classifiers)
	classifier_inds = [list(j) for i in range(1,len(s)) for j in combinations(s, i+1)]

	for index in range(n_learnable):
		grad = []
		for inds in classifier_inds:
			item = inds.copy()
			if index in item:
				item.remove(index)
				temp = np.prod(probs[item, :], axis=0)
				temp_cl = np.prod(1 - probs[item, :], axis=0)

				agreement_item = (temp - temp_cl) / temp.size
				grad.append(agreement_item)
			else:
				grad.append(np.zeros(n))

		gradient.append(grad)
	return np.array(gradient)


def min_agreement_weights(error_rate, w_signal_bounds, data, weights, weak_signal_probabilities):

	"""
	Computes the update for the agreement rate probabilities

    :param error_rate: length n vector of error rates
    :type error_rate: array
    :param w_signal_bounds: list containing the upper error bound of the weak signals
    :type w_signal_bounds: list
    :param data: size (d, n) ndarray containing n examples described by d features each
	:type data: ndarray
	:param weights: matrix containing vectors of weights
	:type weights: ndarray 
	:param weak_signal_probabilities: list containing weak signal probabilities
	:type weak_signal_probabilities: list
    :return: matrix containing optimized weights for the learnable functions
    :rtype: ndarray
    """

	def norm(ndarray, dim=None):
		"""
		:param ndarray: arbitrally vector or matrix of values
		:type ndarray: ndarray
		:param dim: integer indicating which dimension to sum along
		:type dim: int
		:return: l2 norm of vector or 
		:rtype: float
		"""
		return np.sqrt(np.sum(ndarray**2, axis=dim))

	def check_tolerance(vec_t, vec_t2, dim=None):
		"""
		:param vec_t, vec_t: vectors at different timesteps
		:type vector: array
		:param dim: integer indicating which dimension to sum along
		:type dim: int
		:return: boolean value, True if the vectors are equal within a tolerance level
		:rtype: boolean
		"""
		tol = 1e-8
		diff  = norm(vec_t - vec_t2, dim)
		return diff < tol

	probs = probability(data, weights, weak_signal_probabilities)
	agreement_rate = calculate_agreement_rate(probs)
	n_functions = probs.shape[0]

	a_eq, b = Ab_eqmatrix(n_functions, agreement_rate)
	a_ub = Aub_matrix(n_functions)

	lambd = np.zeros(b.size)
	gamma = np.zeros(a_ub.shape[0])
	c, weak_array = initialize_c(w_signal_bounds, n_functions)
	n_learnable = weights.shape[0]
	index = np.where(weak_array > 0)
	error_rate[index] = w_signal_bounds 

	max_iter = 10000
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
		ineq_augmented_term = (a_ub.T).dot(gamma) + (a_ub.T).dot(ue.clip(0,None))
		
		# gradient step update
		e_grad = c + eq_augmented_term + ineq_augmented_term
		error_rate = error_rate - rate * e_grad

		# clip new_error to [0, 1] + weak signals to satisfy constraint
		error_rate[index] = error_rate[index].clip(max=w_signal_bounds, min=0)
		error_rate = np.clip(error_rate, 0, 1)

		# update Lagrange multipliers
		old_lambd = lambd
		lambd = lambd + rate * lambd_grad

		old_gamma = gamma 
		gamma = gamma + rate * ue
		gamma = gamma.clip(0, None)

		# agreement rate gradient
		a_grad = -1 * (old_lambd + rho * lambd_grad) 
		
		# calculate gradient of agreement rate wrt probabilities
		probs_gradient = []
		da_dp = da_dp_gradient(probs, n_learnable)
		# calculate gradient of probabilities
		for i in range(n_learnable): 
			probs_gradient.append(a_grad.T.dot(da_dp[i]))

		# calculate gradient of probabilities wrt weights
		weights_gradient = []
		dp_dw = dp_dw_gradient(data, weights)
		# calculate gradient of weights
		for k in range(n_learnable):
			weights_gradient.append(dp_dw[k].dot(probs_gradient[k]))

		# update weights of the learnable functions
		old_weights = weights
		weights = weights + rate * np.array(weights_gradient)
		
		# calculate new probabilities
		probs = probability(data, weights, weak_signal_probabilities)
		# calculate new agreement rate
		agreement_rate = calculate_agreement_rate(probs)
		# clip new agreement rate to [0, 1] to satisfy constraint
		agreement_rate = np.clip(agreement_rate, 0, 1)

		conv_lambd = norm(np.dot(a_eq, error_rate) - b)
		conv_gamma = norm(np.dot(a_ub, error_rate).clip(min=0))
		conv_error_rate = norm(error_rate - old_error)
		conv_weights = norm(weights - old_weights, dim=1)

		converged = conv_error_rate < 1e-8 and np.isclose(conv_lambd, 0) and \
						np.isclose(conv_gamma, 0) and check_tolerance(weights, old_weights, dim=1).all()


		obj = objective_function(c, error_rate, agreement_rate, rho, lambd, gamma, a_eq, a_ub) # might be slow
		print("Iter %d. Equality infeas: %f, Ineq infeas: %f, Weights Infeas: %f, lagrangian: %f, obj: %f" % (t, conv_lambd, conv_gamma, 
																								np.sum(conv_weights), obj, error_rate.dot(c)))

		t += 1

	print("No of learnable classifiers is %d and total functions is %d" %(n_learnable, n_functions))
	# hack to plot objective around current error_rate
	vector = np.random.randn(error_rate.size)
	eps = np.linspace(-1, 1, 100)
	obj_val = np.zeros(eps.size)

	for j in range(eps.size):
		obj_val[j] = objective_function(c, error_rate + eps[j] * vector, agreement_rate, rho, lambd, gamma, a_eq, a_ub)

	plt.plot(eps, obj_val)
	plt.show()

	eq_cons = np.dot(a_eq, error_rate)
	ineq_cons = np.dot(a_ub, error_rate)


	print('Estimated error rate:', error_rate)
	print('Estimated agreement_rate:', agreement_rate)
	# print('Optimized weights:', weights)
	# print("Optimized probabilities", probability(data, weights, weak_signal_probabilities))
	#print("Equality infeasibility:", eq_cons, b)
	#print("Inequality infeasibily:" + repr(ineq_cons))

	#assert np.allclose(eq_cons, b)
	ineq_cons[ineq_cons < 0] = 0
	#assert np.allclose(ineq_cons, 0)
	return weights


def initialize_c(w_signal_bound, n_functions):
	"""
    :param w_signal_bound: vector containing the upper bound error rate values of the weak signals 
    :type w_signal_bound: array
    :param n_functions: number of functions
    :type n_functions: integer
    :return: tuple of vector of c and upper bound of weak_signals
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