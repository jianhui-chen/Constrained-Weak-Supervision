import numpy as np
import matplotlib.pyplot as plt
#from setup_supervision import accuracy_score, writeToFile
from setup_model import accuracy_score, writeToFile, mlp_model
from scipy.optimize import check_grad
import random, json, sys, gc
#from utilities import projection_simplex
from data_utilities import projection_simplex
from tensorflow.python.keras import backend as K
#from setup_model import convnet_model


def multiclass_loss(y, learnable_probabilities):
    objective = y * (1 - learnable_probabilities)
    gradient = 1 - learnable_probabilities
    return np.sum(objective) / y.shape[0], gradient


def multilabel_loss(y, learnable_probabilities):
    objective = y * (1 - learnable_probabilities) + learnable_probabilities * (1 - y)
    n,k = y.shape
    gradient = (1 - 2 * learnable_probabilities) /(n*k)
    return np.mean(objective), gradient


def quadratic_loss(y, learnable_probabilities):
    objective = (y - learnable_probabilities)**2
    n,k = y.shape
    gradient = 0.5*(y - learnable_probabilities)/(n*k)
    return np.mean(objective), gradient


def crossentropy_loss(y, learnable_probabilities):
    objective = -y * np.log(learnable_probabilities+1e-8)
    n,k = y.shape
    gradient = -np.log(learnable_probabilities+1e-8)/(n*k)
    return np.mean(objective), gradient


def loss_functions(y, learnable_probabilities, loss='crossentropy'):

    if loss == 'multiclass':
        return multiclass_loss(y, learnable_probabilities)

    if loss == 'multilabel':
        return multilabel_loss(y, learnable_probabilities)

    if loss == 'quadratic':
        return multilabel_loss(y, learnable_probabilities)

    return crossentropy_loss(y, learnable_probabilities)


def objective_function(y, learnable_probabilities, constraint_set, rho):
    """
    Computes the value of the objective function
    One weak signal contains k num of weak signals, one for each class k=num_class

    :param y: size (n_data_points, num_class) adversarial labels for the data
    :type y: ndarray
    :param learnable_probabilities: size (n_data_points, num_class) of estimated probabilities for the learnable classifier
    :type learnable_probabilities: ndarray
    :param constraint_set: dictionary containing constraints specified in the constraint_keys
    :type constraint_set: dict
    :param rho: penalty parameter for the augmented lagrangian
    :type rho: float
    :return: scalar value of objective function
    :rtype: float
    """

    gamma_constraint = 0
    augmented_term = 0
    constraint_keys = constraint_set['constraints']
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']

    objective = loss_functions(y, learnable_probabilities, loss)

    for key in constraint_keys:
        current_constraint = constraint_set[key]
        a_matrix = current_constraint['A']
        bounds = current_constraint['b']
        constant = current_constraint['c']
        gamma = current_constraint['gamma']
        constraint = np.zeros(bounds.shape)

        for i, current_a in enumerate(a_matrix):
            # constraint[i] = np.sum((active_mask[i]*current_a) * y + (constant[i]*active_mask[i]), axis=0)
            constraint[i] = np.sum(current_a * y + constant[i], axis=0)

        constraint = constraint - bounds

        gamma_constraint += np.sum(gamma * constraint)
        augmented_term += (rho / 2) * np.sum(
            constraint.clip(min=0) * constraint.clip(min=0))

    return objective + gamma_constraint - augmented_term


#def bound_loss(y, a_matrix, active_mask, constant, bounds):
def bound_loss(y, a_matrix, constant, bounds):
    """
    Computes the gradient of lagrangian inequality penalty parameters

    :param y: size (n_data_points, num_class) of estimated labels for the data
    :type y: ndarray
    :param a_matrix: size (num_weak, n, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param constant: size (num_weak, n, num_class) of the constant
    :type constant: ndarray
    :param bounds: size (num_weak, num_class) of the bounds for the constraint
    :type bounds: ndarray
    :return: loss of the constraint set wrt adversarial ys
    :rtype: ndarray
    """
    constraint = np.zeros(bounds.shape)
    n, k = y.shape

    for i, current_a in enumerate(a_matrix):
        #constraint[i] = np.sum((active_mask[i]*current_a) * y + (constant[i]*active_mask[i]), axis=0)
        constraint[i] = np.sum(current_a * y + constant[i], axis=0)
    return constraint - bounds


def y_gradient(learnable_probabilities, constraint_set, rho, y, quadratic=False):
    """
    Computes the gradient y

    :param learnable_probabilities: size (n_data_points, num_class) of probabilities for the learnable classifier
    :type learnable_probabilities: ndarray
    :param a_matrix: size (num_weak, n, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param gamma: vector of lagrangian inequality(upper bound) penalty parameters corresponding to the number of weak signals
    :type gamma: array
    :param rho: penalty parameter for the augmented lagrangian
    :type rho: float
    :return: ndarray of size (n_data_points, num_class) for y gradient
    :rtype: ndarray
    """

    n, k = learnable_probabilities.shape
    augmented_term = 0
    upper_bound_term = 0
    constraint_keys = constraint_set['constraints']
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']

    obj_grad = 1 - learnable_probabilities \
                    if loss == 'multiclass' else 1 - 2*learnable_probabilities

    obj_grad = obj_grad / n if loss == 'multiclass' else obj_grad / (n*k)


    for key in constraint_keys:
        current_constraint = constraint_set[key]
        a_matrix = current_constraint['A']
        bound_loss = current_constraint['bound_loss']
        gamma = current_constraint['gamma']

        for i, current_a in enumerate(a_matrix):
            #constraint = a_matrix[i] * active_mask[i]
            constraint = a_matrix[i]
            upper_bound_term += gamma[i] * constraint
            augmented_term += bound_loss[i].clip(min=0) * constraint

    return obj_grad + upper_bound_term - rho * augmented_term


def run_constraints(label, predicted_probs, rho, constraint_set, iters=300, enable_print=True, optim='min'):
    # First find a feasible label with adagrad, initialization step

    constraint_keys = constraint_set['constraints']
    weak_signals = constraint_set['weak_signals']
    num_weak_signal = constraint_set['num_weak_signals']
    # true_bounds = constraint_set['true_bounds'] # boolean value
    # true_bounds = False
    true_bounds = True
    loss = constraint_set['loss']
    #active_mask = constraint_set['active_mask']
   
    grad_sum = 0
    y = label.copy()
    n,k = y.shape

    #weak_signals = weak_signals * active_mask

    # get the min weak_signal vector
    min_vector = np.min(weak_signals[:num_weak_signal, :, :], axis=0)
    max_vector = np.max(weak_signals[:num_weak_signal, :, :], axis=0)
    assert y.shape == min_vector.shape

    for iter in range(iters):
        print_constraints = [iter]
        print_builder = "Iteration %d, "
        constraint_viol = []
        viol_text = ''

        for key in constraint_keys:

            current_constraint = constraint_set[key]

            # print(constraint_keys)
            # print(current_constraint)
            # exit()

            a_matrix = current_constraint['A']
            bounds = current_constraint['b']
            constant = current_constraint['c']
            gamma = current_constraint['gamma']

            # print(a_matrix.shape)
            # exit()

            # get bound loss for constraint
            #full_loss = bound_loss(y, a_matrix, active_mask, constant, bounds)
            full_loss = bound_loss(y, a_matrix, constant, bounds)
            if iter == 0 or iter == (iters - 1):
                print("full loss")
                print(full_loss)
            gamma_grad = full_loss
            
            if iter == 0 or iter == (iters - 1):
                print("Gamma pre ")
                print(gamma)
            if optim == 'max':
                if iter == 0 or iter == (iters - 1):
                    print("max")
                gamma = gamma - rho * gamma_grad

                if iter == 0 or iter == (iters - 1):
                    print("Gamma pre clip")
                    print(gamma)
                gamma = gamma.clip(max=0)
                if iter == 0 or iter == (iters - 1):
                    print("Gamma postclip ")
                    print(gamma)
            else:
                if iter == 0 or iter == (iters - 1):
                    print("min")
                gamma = gamma + rho * gamma_grad
                if iter == 0 or iter == (iters - 1):
                    print("Gamma pre clip")
                    print(gamma)
                gamma = gamma.clip(min=0)
                if iter == 0 or iter == (iters - 1):
                    print("Gamma postclip ")
                    print(gamma)

           

            # update constraint values
            constraint_set[key]['gamma'] = gamma
            constraint_set[key]['bound_loss'] = full_loss

            violation = np.linalg.norm(full_loss.clip(min=0))
            print_builder += key + "_viol: %.4e "
            print_constraints.append(violation)

            viol_text += key + "_viol: %.4e "
            constraint_viol.append(violation)

        y_grad = y_gradient(predicted_probs, constraint_set, rho, y, quadratic=True)
        grad_sum += y_grad**2

        if iter == 0 or iter == (iters - 1):
            print("y grad and grad sum and y")
            print(y_grad)
            print(grad_sum)
            print(y)

        if optim == 'max':
            y = y + y_grad / np.sqrt(grad_sum + 1e-8)
        else:
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)

        if iter == 0 or iter == (iters - 1):
            print("y post calc update")
            print(y)

        # Commenting out just to see y values
        y = np.clip(y, a_min=min_vector, a_max=max_vector)  if not true_bounds \
                                else (y if loss == 'multiclass' else np.clip(y, a_min=0, a_max=1))
        y = projection_simplex(y, axis=1) if loss == 'multiclass' else y

        if iter == 0 or iter == (iters - 1):
            print("y post clip code")
            print(y)

        constraint_set['violation'] = [viol_text, constraint_viol]
        # if enable_print:
        #     print(print_builder % tuple(print_constraints))

    print(np.count_nonzero(y<.5))

    return y, constraint_set



def train_stochgall(data_info, constraint_set, max_epoch=20):
    """
    Trains the cnn model for image classification NOPE NOT ANYMORE

    :param labels: True labels for the data, only used for debugging
    :type labels: ndarray
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param constraint_set: dictionary containing constraints specified in the constraint_keys
    :type constraint_set: dict
    :return: dictionary of results
    :rtype: dict
    """

    data = data_info['train_data']
    labels = data_info['train_labels']
    test_data, test_labels = data_info['test_data'], data_info['test_labels']


    # why image
    """
    img_rows, img_cols = data_info['img_rows'], data_info['img_cols']
    channels = data_info['channels']
    """

    num_weak_signal = data_info['num_weak_signals']
    constraint_keys = constraint_set['constraints']
    weak_signals = constraint_set['weak_signals']

    # print(weak_signals)
    # print(weak_signals.shape)
    # exit()

    # What is loss and optim
    loss = constraint_set['loss']
    #optim = constraint_set['optim']

    results = dict()
    m, n, k = weak_signals.shape

    ## THIS IS FOR COMPARING ACCURACY LATER 
    if k==1:
       labels = labels.reshape(n,k)

    m = 2 if k==1 else k
    # remove if not random p
    # learnable_probabilities = np.random.rand(n,k)
    learnable_probabilities = np.ones((n,k)) * 1/m
    # learnable_probabilities = labels.copy()

    # initialize y
    # y = labels.copy()
    y = np.ones((n, k)) * 0.1
    assert y.shape == labels.shape

    # initialize hyperparameters
    rho = 0.1
    batch_size = 32


    print("Initial running")
    # This is to prevent the learning algo from wasting effort fitting a model to arbitrary y values.
    y, constraint_set = run_constraints(y, learnable_probabilities, rho, constraint_set, optim='max')

    # Train Stoch-gall
    #model = covnet_model(img_rows, img_cols, channels, loss)

    #MLP model here         IS THIS THE CORRECT FORM
    # print(k)
    # print(data.shape)
    # print(n)
    # print(m)
    model = mlp_model(data.shape[1], k)
    # exit()

    print("Running adversarial label learning..")
    grad_sum = 0
    epoch = 0
    while epoch < max_epoch:
        indices = list(range(n))
        random.shuffle(indices)
        batches = np.array_split(indices, int(n / batch_size))

        rate = 1.0  # constant step_size
        # rate = 1 / np.sqrt(1 + t) # decreasing step size
        old_y = y.copy()

        print_constraints = [epoch]
        print_builder = "Epoch %d, "


        if epoch % 1 == 0:
            for batch in batches:
                model.train_on_batch(data[batch], y[batch])
            learnable_probabilities = model.predict(data)

        print("learnable probs ")
        print(learnable_probabilities)
        # exit()

        if epoch % 2 == 0:
            y, constraint_set = run_constraints(y, learnable_probabilities, rho, constraint_set, iters=10, enable_print=False)
            print_builder += constraint_set['violation'][0]
            print_constraints.extend(constraint_set['violation'][1])

            # exit()


            

        # if ((y==old_y).all()):
        #     print("same")
        #     print(y)
        #     exit()

        # calculate change in y
        # change_y = np.linalg.norm(y - old_y)
        # print_constraints.append(change_y)

        # objective = multiclass_loss(y, learnable_probabilities)[0] \
        #             if loss == 'multiclass' else multilabel_loss(y, learnable_probabilities)[0]
        # print_constraints.append(objective)

        # lagrangian_obj = objective_function(y, learnable_probabilities, constraint_set, rho)  # might be slow
        # print_constraints.append(lagrangian_obj)

        epoch += 1

        # print_builder += "delta_y: %.4f, obj: %.4f, lagr: %.4f "
        print(print_builder % tuple(print_constraints))

    print("")
    # print(y) For some reason, everything in 1

    label_accuracy = accuracy_score(labels, y)

    y_pred = model.predict(data)
    train_accuracy = accuracy_score(labels, y_pred)

    # calculate test results
    y_pred = model.predict(test_data)
    test_accuracy = accuracy_score(test_labels, y_pred)

    print('label acc: %f' %(label_accuracy))

    print('Stoch-gall train_acc: %f, test_accu: %f' %(train_accuracy, test_accuracy))

    results['label_accuracy'] = label_accuracy
    results["test_accuracy"] = test_accuracy
    results['train_accuracy'] = train_accuracy

    K.clear_session()
    del model
    gc.collect()

    # print("Saving to file...")
    # filename = 'results/new_results/stoch-gall_results.json'
    # writeToFile(results, filename)

    #need to return something
    exit()
    return results

