from real_experiments import run_experiment, bound_experiment, dependent_error_exp
from data_readers import *
from classes_file import Data
from weak_signals import *
from new_loop import *


def run_tests():
    """
    Runs experiment.
    :return: None
    """
    
    # # # # # # # # # # # #
    # breast cancer       #
    # # # # # # # # # # # #

    #for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
    print("\n\n# # # # # # # # # # # # # # # # # # # # #")
    print("# Running breast cancer experiment...   #")
    print("# # # # # # # # # # # # # # # # # # # # #\n")
    bc_data = Data("Breast Cancer", [0, 10, 20], 'datasets/breast-cancer/wdbc.data', 'results/json/breast_cancer.json', breast_cancer_load_and_process_data)
    w_data_dicts = get_w_data_dicts(bc_data, 1, 3)
    # adversarial_acc_dicts, w_acc_dicts = run_experiment(bc_data, w_data_dicts)
    new_run_experiment(bc_data, w_data_dicts)


    
    # # # # # # # # # # # # #
    # # obs network         #
    # # # # # # # # # # # # #

    # # #for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals
    # print("\n\n\n\n# # # # # # # # # # # # # # # # # # # # #")
    # print("# Running obs network experiment...     #")
    # print("# # # # # # # # # # # # # # # # # # # # #\n")


    # obs_data = Data("OBS", [1, 2, 20], 'datasets/obs-network/obs_network.data', 'results/json/obs_network.json', obs_load_and_process_data)
    # w_data_dicts = get_w_data_dicts(obs_data, 1, 3)
    # adversarial_acc_dicts, w_acc_dicts = run_experiment(obs_data, w_data_dicts)


    # # # # # # # # # # # # #
    # # cardio              #
    # # # # # # # # # # # # #
 
    # # #Use AC, MLTV and Median as weak signal views
    # print("\n\n\n\n# # # # # # # # # # # # # # # # # # # # #")
    # print("# Running cardio experiment...          #")
    # print("# # # # # # # # # # # # # # # # # # # # #\n")

    # cardio_data = Data("Cardio", [1, 10, 18], 'datasets/cardiotocography/cardio.csv', 'results/json/cardio.json', cardio_load_and_process_data)
    # w_data_dicts = get_w_data_dicts(cardio_data, 1, 3)
    # adversarial_acc_dicts, w_acc_dicts = run_experiment(cardio_data, w_data_dicts)
    

def run_bounds_experiment():
    """
    Runs experiment.
    :return: None
    """

    # # # # # # # # # # # #
    # breast cancer       #
    # # # # # # # # # # # #
    
    # for breast cancer classification dataset, select the mean radius, radius se and worst radius as weak signals
   
    print("\n\nRunning bounds on breast cancer experiment...")

    bc_data  = Data("Breast Cancer", [0, 10, 20], 'datasets/breast-cancer/wdbc.data', 'results/json/bc_bounds.json', breast_cancer_load_and_process_data)
    w_data_dicts = get_w_data_dicts(bc_data, 3, 3)

    bound_experiment(bc_data, w_data_dicts[0])


    # # # # # # # # # # # #
    # obs network         #
    # # # # # # # # # # # #

    # for obs network dataset, select the Utilized Bandwidth Rate, Packet drop rate and Flood Status as weak signals

    print("\n\nRunning bounds on obs network experiment...")

    obs_data = Data("OBS", [1, 2, 20], 'datasets/obs-network/obs_network.data', 'results/json/obs_bounds.json', obs_load_and_process_data)
    w_data_dicts = get_w_data_dicts(obs_data, 3, 3)

    bound_experiment(obs_data, w_data_dicts[0])




def run_dep_err_experiment():

    # # # # # # # # # # # #
    # cardio              #
    # # # # # # # # # # # #

    print("\n\nRunning dependent error on cardio experiment...")
     
    #Use AC, MLTV and Median as weak signal views, and repeat the bad weak signal 
    views       = [1, 18, 18, 18, 18, 18, 18, 18, 18, 18]
    cardio_data = Data("Cardio", views, 'datasets/cardiotocography/cardio.csv', 'results/json/cardio_error.json', cardio_load_and_process_data)

    w_data_dicts = get_w_data_dicts(cardio_data, 1, 10)

    dependent_error_exp(cardio_data, w_data_dicts)


if __name__ == '__main__':
    run_tests()

    # # un-comment to run bounds experimrnt in the paper
    # run_bounds_experiment()

    # # un-comment to run dependency error experiment in the paper
    # run_dep_err_experiment()