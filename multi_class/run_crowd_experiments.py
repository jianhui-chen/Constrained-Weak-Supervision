import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from setup_model import mlp_model, accuracy_score, prepare_mmce, writeToFile
from mmce import MinimaxEntropy_crowd_model

ROOT_DIR = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIR)
sys.path.append('./data_consistency')

from consistency_experiments import consistency_data
from train_classifier import train_all
from train_stochastic import train_stochastic_all

def prepare_bluebird_data():
    datapath = '../datasets/bluebirds/'
    df = pd.read_csv(datapath+'original.tsv', sep='\t', header=None)
    true_labels = df.iloc[:,3].values
    workers = np.unique(df.iloc[:,0])
    instance_ids = np.unique(df.iloc[:,1])

    crowd_labels = []
    for workid in workers:
        worker_df = df.loc[df[0] == workid]
        worker_labels = worker_df.iloc[:,2].values
        assert (instance_ids == worker_df.iloc[:,1].values).all()
        crowd_labels.append(worker_labels)

    n = worker_df.shape[0]
    crowd_labels = np.asarray(crowd_labels).T
    true_labels = true_labels[:n]
    assert true_labels.size == crowd_labels.shape[0]

    # save the crowd & true labels
    np.save(datapath+'crowd_labels.npy', crowd_labels)
    np.save(datapath+'true_labels.npy', true_labels)

    df = pd.read_csv(datapath+'features.txt', sep="\t", quoting=3, header=None)
    data_features = []
    for index, row in df.iterrows():
        data_features.append(row[1].split(' '))
    data_features = np.asarray(data_features).astype(float)
    np.save(datapath+'data_features.npy', data_features)


def prepare_medical_relations_data(name='causes'):
    datapath = '../datasets/medical-relations/'
    df = pd.read_csv(datapath+'original_treats.tsv', sep='\t', header=None, names=['worker','instance','label','predictor'])
    df = df.drop_duplicates()
    workers = df['worker'].unique()
    instances = np.sort(df['instance'].unique())

    crowd_labels = []
    true_labels = []
    for instid in instances:
        inst_df = df.loc[df['instance'] == instid]
        labels = []
        for workid in workers:
            worker_df = inst_df.loc[inst_df['worker'] == workid].drop_duplicates(subset=['worker'])
            if worker_df.empty:
                labels.append(-1)
                continue
            assert worker_df.shape[0] == 1
            labels.append(worker_df['predictor'].values[0])
        crowd_labels.append(labels)
        true_labels.append(inst_df['label'].values[0])

    crowd_labels = np.asarray(crowd_labels)
    true_labels = np.asarray(true_labels)
    assert true_labels.size == crowd_labels.shape[0]

    # save the crowd & true labels
    np.save(datapath+name+'/crowd_labels.npy', crowd_labels)
    np.save(datapath+name+'/true_labels.npy', true_labels)

    df = pd.read_csv(datapath+'features.txt', sep=" ", quoting=3, header=None)
    data_features = df.values
    assert true_labels.size == data_features.shape[0]
    np.save(datapath+name+'/data_features.npy', data_features)


def prepare_text_data(datapath):
    # datapath = '../datasets/wordsim/'
    name = datapath.split('/')[-2]
    df = pd.read_csv(datapath+'original.tsv', sep='\t')
    instances = np.unique(df['orig_id'])
    crowd_labels = []
    true_labels = []

    for instid in instances:
        inst_df = df.loc[df['orig_id'] == instid]
        worker_labels = inst_df['response'].values
        worker_labels = worker_labels/10 if name=='wordsim' else worker_labels
        crowd_labels.append(worker_labels)
        true_labels.append(inst_df['gold'].values[0])

    crowd_labels = np.round(crowd_labels)
    true_labels = np.asarray(true_labels)
    if name=='wordsim':
        true_labels[true_labels < 2.5] = 0
        true_labels[true_labels > 2.5] = 1
    assert true_labels.size == crowd_labels.shape[0]

    # save the crowd & true labels
    np.save(datapath+'crowd_labels.npy', crowd_labels)
    np.save(datapath+'true_labels.npy', true_labels)

    df = pd.read_csv(datapath+'features.txt', sep="\t", quoting=3, header=None)
    data_features = []
    for instid in instances:
        inst_df = df.loc[df[0] == instid]
        data_features.append(inst_df[1].values[0].split(' '))
    data_features = np.asarray(data_features).astype(float)
    np.save(datapath+'data_features.npy', data_features)


def run_red_experiments(datapath, savename):
    train_data = np.load(datapath+'data_features.npy', allow_pickle=True)[()]
    crowd_labels = np.load(datapath+'crowd_labels.npy', allow_pickle=True)[()].astype(int)
    train_labels = np.load(datapath+'true_labels.npy', allow_pickle=True)[()]
    weak_signals = crowd_labels.copy()

    ## Run MMCE and majority vote experiments
    # mmce_crowd_labels, transformed_labels = prepare_mmce(weak_signals, train_labels)
    # mmce_labels, mmce_error_rate = MinimaxEntropy_crowd_model(mmce_crowd_labels, transformed_labels)

    ## for RTE experiments
    # dataset = {}
    # dataset['train'] = train_data, train_labels
    # dataset['test'] = X_test, y_test
    # embedding = consistency_data(dataset, form='embedding', encoding_dim=1000, no_clusters=80).numpy()

    ### Run ALL experiment
    ## bound = np.ones(crowd_labels.shape[1]) * 0.1 # constant bounds
    bound = 1 - np.mean(weak_signals.T == train_labels, axis=1) # true labels
    learned_labels, optimized_weights = train_all(train_data.T, weak_signals.T, bound, max_iter=4000)
    # learned_labels, optimized_weights = train_stochastic_all(embedding, weak_signals.T, bound, max_epochs=50)
    all_accuracy = accuracy_score(train_labels, np.round(learned_labels))
    print("ALL accuracy is: ", all_accuracy)

    # results = {}
    # filename = 'data_consistency/results/all_crowd_results.json'
    # results['all_accuracy'] = all_accuracy
    # results['experiment'] = savename

    # with open(filename, 'a') as file:
    #     json.dump(results, file, indent=4, separators=(',', ':'))
    # file.close()


def run_experiments(datapath):
    train_data = np.load(datapath+'data_features.npy', allow_pickle=True)[()]
    crowd_labels = np.load(datapath+'crowd_labels.npy', allow_pickle=True)[()].astype(int)
    train_labels = np.load(datapath+'true_labels.npy', allow_pickle=True)[()]

    savedir = 'results/'
    all_accs =[]
    mmce_accs=[]
    mv_accs=[]
    for i in range(1, 11):
        alls = []
        mmces =[]
        mvs=[]
        for j in range(1):
            indices = np.random.randint(10, size=i)
            weak_signals = crowd_labels.T[:i]
            weak_signals = weak_signals.T
            bound = 1 - np.mean(weak_signals.T == train_labels, axis=1)
            # bound = np.ones(weak_signals.shape[1]) * 0.3
            # Run ALL experiment
            # learned_labels, optimized_weights = train_all(train_data.T, weak_signals.T, bound, max_iter=4000)
            learned_labels, optimized_weights = train_stochastic_all(train_data, weak_signals.T, bound, max_epochs=50)

            # Run MMCE experiments
            mmce_crowd_labels, transformed_labels = prepare_mmce(weak_signals, train_labels)
            mmce_labels, mmce_error_rate = MinimaxEntropy_crowd_model(mmce_crowd_labels, transformed_labels)
            majority_vote_labels = np.mean(weak_signals, axis=1)
            break_ties = np.random.randint(2, size=int(np.sum(majority_vote_labels==0.5)))
            majority_vote_labels[majority_vote_labels==0.5] = break_ties

            all_accuracy = accuracy_score(train_labels, np.round(learned_labels))
            mmce_accuracy = 1 - mmce_error_rate
            mv_accuracy = accuracy_score(train_labels, np.round(majority_vote_labels))

            alls.append(all_accuracy)
            mmces.append(mmce_accuracy)
            mvs.append(mv_accuracy)

        all_accs.append(np.mean(alls))
        mmce_accs.append(np.mean(mmces))
        mv_accs.append(np.mean(mvs))
        print("All label accuracy is:", all_accuracy)
        print("MMCE label accuracy is:", mmce_accuracy)
        print("Majority vote accuracy is:", mv_accuracy)

    output = {}
    output['ALL'] = all_accs
    output['MMCE'] = mmce_accs
    output['MV'] = mv_accs
    writeToFile(output, 'results/all_good_signals.json')

    weak_signals = np.arange(10) + 1

    plt.figure(figsize=[4, 2.5])
    plt.plot(weak_signals, all_accs, ':.', color="b", label='ALL')
    plt.plot(weak_signals, mmce_accs, ':.', color="r", label='MMCE')
    plt.plot(weak_signals, mv_accs, ':.', color="g", label='Majority Vote')
    plt.xlabel('Weak signals')
    plt.ylabel('Accuracy')
    # plt.ylim(ymin=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/good_signal_results.pdf', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':

    # print("Running experiments on medical relations datasets:")
    # run_red_experiments("../datasets/rte/", 'rte')
    run_red_experiments("../datasets/wordsim/", 'wordsim')
    run_red_experiments("../datasets/bluebirds/", 'bluebirds')
    # run_red_experiments("../datasets/medical-relations/treats/",'medical-treats')
    # run_red_experiments("../datasets/medical-relations/causes/", 'medical-causes')
    ################################################################

    # print("Running experiments on crowd source datasets...")
    # print("Running experiments on rte datasets:")
    # run_experiments("../datasets/rte/")
    # print("Running experiments on wordsim datasets:")
    # run_experiments("../datasets/wordsim/")
    # print("Running experiments on bluebirds datasets:")
    # run_experiments("../datasets/bluebirds/")
    # print("Running experiments on medical relations datasets:")
    # run_experiments("../datasets/medical-relations/treats/")
