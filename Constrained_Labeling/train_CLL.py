import numpy as np


def bound_loss(y, a_matrix, bounds):
    """
    Computes the gradient of lagrangian inequality penalty parameters

    :param y: size (num_data, num_class) of estimated labels for the data
    :type y: ndarray
    :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param bounds: size (num_weak, num_class) of the bounds for the constraint
    :type bounds: ndarray
    :return: loss of the constraint (num_weak, num_class)
    :rtype: ndarray
    """
    constraint = np.zeros(bounds.shape)
    n, k = y.shape

    for i, current_a in enumerate(a_matrix):
        constraint[i] = np.sum(current_a * y, axis=0)
    return constraint - bounds


def y_gradient(y, constraint_set):
    """
    Computes y gradient
    """
    constraint_keys = constraint_set['constraints']
    gradient = 0

    for key in constraint_keys:

        print("hello")
        current_constraint = constraint_set[key]
        a_matrix = current_constraint['A']
        bound_loss = current_constraint['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]


    return gradient


def run_constraints(y, rho, constraint_set, iters=300, enable_print=True):
    # Run constraints from CLL

    constraint_keys = constraint_set['constraints']
    n, k = y.shape
    rho = n
    grad_sum = 0
    lamdas_sum = 0

    # debugging code
    # print("\n\n y:", y) 
    # print("y type:", type(y))
    # print("y shape:", y.shape,"\n\n")
    # print("\n\n constraint_keys:", constraint_keys) 
    # print("constraint_keys type:", type(constraint_keys), "\n\n")
    # print("\n\n constraint_keys:", constraint_set['error']) 
    # print("constraint_keys type:", type(constraint_set['error']), "\n\n")


    for iter in range(iters):
        # print_constraints = [iter]
        # print_builder = "Iteration %d, "
        # constraint_viol = []
        # viol_text = ''

        # current_constraint = constraint_set['error']
        a_matrix = constraint_set['error']['A']
        bounds = constraint_set['error']['b']


        # print("\n\n a_matrix:", a_matrix) 
        # print("a_matrix type:", type(a_matrix))
        # print("a_matrix shape:", a_matrix.shape,"\n\n")
        # print("\n\n bounds:", bounds) 
        # print("bounds type:", type(bounds))
        # print("bounds shape:", bounds.shape,"\n\n")

        loss = bound_loss(y, a_matrix, bounds)

        # update constraint values
        constraint_set['error']['bound_loss'] = loss

        # for key in constraint_keys:

        #     # print("\n\n y:", key) 
        #     # print("y type:", type(key),"\n\n")
            

        #     current_constraint = constraint_set[key]
        #     a_matrix = current_constraint['A']
        #     bounds = current_constraint['b']

        #     # get bound loss for constraint
        #     loss = bound_loss(y, a_matrix, bounds)
        #     # update constraint values
        #     constraint_set[key]['bound_loss'] = loss

        #     # violation = np.linalg.norm(loss.clip(min=0))
        #     # print_builder += key + "_viol: %.4e "
        #     # print_constraints.append(violation)

        #     # viol_text += key + "_viol: %.4e "
        #     # constraint_viol.append(violation)

        y_grad = y_gradient(y, constraint_set)
        grad_sum += y_grad**2
        y = y - y_grad / np.sqrt(grad_sum + 1e-8)
        y = np.clip(y, a_min=0, a_max=1)

        # constraint_set['violation'] = [viol_text, constraint_viol]
        # if enable_print:
        #     print(print_builder % tuple(print_constraints))
    return y


def train_algorithm(constraint_set):
    """
    Trains CLL algorithm

    :param constraint_set: dictionary containing error constraints of the weak signals
    :return: average of learned labels over several trials
    :rtype: ndarray
    """
    constraint_set['constraints'] = ['error']
    weak_signals = constraint_set['weak_signals']
    assert len(weak_signals.shape) == 3, "Reshape weak signals to num_weak x num_data x num_class"
    m, n, k = weak_signals.shape
    # initialize y
    y = np.random.rand(n, k)
    # initialize hyperparameters
    rho = 0.1

    t = 3  # number of random trials
    ys = []
    for i in range(t):
        ys.append(run_constraints(y, rho, constraint_set))
    return np.mean(ys, axis=0)
