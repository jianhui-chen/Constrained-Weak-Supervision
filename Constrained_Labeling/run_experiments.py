import numpy as np
from model_utilities import majority_vote_signal, mlp_model, set_up_constraint, get_error_bounds
from data_readers import read_text_data
from models import CLL

def generate_synthetic_data():
    """ 
        Generates synthetic data

        Parameters
        ----------
        none

        Returns
        -------
        :returns: training set, testing set, and weak signals 
                  of synthetic data
        :return type: dictionary of ndarrays
    """

    np.random.seed(900)
    n  = 20000
    d  = 200
    m = 10
    Ys = 2 * np.random.randint(2, size=(n,)) - 1

    feature_accs = 0.2 * np.random.random((d,)) + 0.5
    train_data = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            if np.random.random() > feature_accs[j]:
                train_data[i,j] = 1 if Ys[i] == 1 else 0
            else:
                train_data[i,j] = 0 if Ys[i] == 1 else 1


    # Initialize the weak signals
    ws_accs = 0.1 * np.random.random((m,)) + 0.6
    ws_coverage = 0.3
    Ws = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if np.random.random() < ws_coverage:
                Ws[i,j] = Ys[i] if np.random.random() < ws_accs[j] else -Ys[i]

    # Convert weak_signals to correct format
    weak_signals = Ws.copy()
    weak_signals[Ws==0] = -1
    weak_signals[Ws==-1] = 0

    n,m = weak_signals.shape
    weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    # Convert Y and weak_signals to correct format
    train_labels = 0.5 * (Ys + 1)

    indexes = np.arange(n)
    np.random.seed(2000)
    test_indexes = np.random.choice(n, int(n * 0.2), replace=False)
    weak_signals = np.delete(weak_signals, test_indexes, axis=1)

    test_labels = train_labels[test_indexes]
    test_data = train_data[test_indexes]
    train_indexes = np.delete(indexes, test_indexes)
    train_labels = train_labels[train_indexes]
    train_data = train_data[train_indexes]

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals

    return data


def run_experiment(dataset, true_bound=False):
    """ 
        Run CLL experiments on real data

        Parameters
        ----------
        :param dataset: training set, testing set, and weak signals 
                        of dataset
        :type  dataset: dictionary of ndarrays
        :param true_bound: determinds wether errors are random or 
                           based on training labels or 
        :type  true_bound: boolean

        Returns
        -------
        nothing
    """


    print("\nrunning expirements\n")

    # Set up variables
    train_data, train_labels  = dataset['train']
    test_data, test_labels    = dataset['test']
    weak_signals              = dataset['weak_signals']
    m, n, k                   = weak_signals.shape

    current_CLL = CLL(log_name="Label_Estimator ")
    current_mlp = mlp_model(train_data.shape[1], k)

    # Set up the error bounds 
    weak_errors = np.ones((m, k)) * 0.01
    if true_bound:
        weak_errors = get_error_bounds(train_labels, weak_signals)
        weak_errors = np.asarray(weak_errors)
    error_set     = set_up_constraint(weak_signals, weak_errors)

    # run CLL to estimate labels
    y             = current_CLL.fit(weak_signals, error_set)
    accuracy      = current_CLL.get_accuracy(train_labels, y)

    # Use estimated labels to train a new algorithm
    current_mlp.fit(train_data, y, batch_size=32, epochs=20, verbose=1)
    test_predictions = current_mlp.predict(test_data)
    test_accuracy    = current_CLL.get_accuracy(test_labels, test_predictions)

    print("CLL Label accuracy is: ", accuracy)
    print("CLL Test accuracy is: \n", test_accuracy)

    # Compare against majority vote expirment 
    mv_labels = majority_vote_signal(weak_signals)
    print("Majority vote accuracy is: ", current_CLL.get_accuracy(train_labels, mv_labels))






# # Expiriments:
# # ------------

# print("\n\n# # # # # # # # # # # # # # # #")
# print("#  synthetic data experiment  #")
# print("# # # # # # # # # # # # # # # #\n")
# run_experiment(generate_synthetic_data())


# print("\n\n# # # # # # # # # # # #")
# print("#  sst-2  experiment  #")
# print("# # # # # # # # # # # #\n")
# run_experiment(read_text_data('../datasets/sst-2/'))

print("\n\n# # # # # # # # # # #")
print("#  imdb experiment  #")
print("# # # # # # # # # # #\n")
run_experiment(read_text_data('../datasets/imdb/'))

# # NOT READY YET:
# # ------------

# print("\n\n# # # # # # # # # # # #")
# print("#   yelp experiment   #")
# print("# # # # # # # # # # # #\n")
# print("\nyelp experiment\n")
# run_experiment(read_text_data('../datasets/yelp/'))
