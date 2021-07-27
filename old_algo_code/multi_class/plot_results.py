import numpy as np
import matplotlib.pyplot as plt
from json import JSONDecoder
import json, codecs
from functools import partial

def load_json_data(filename):

    with open(filename) as json_file:
        data = json.load(json_file)
    return data

# custom method to load snorkel results
def json_parse(fileobj, decoder=JSONDecoder(), buffersize=2048):
    buffer = ''
    for chunk in iter(partial(fileobj.read, buffersize), ''):
        buffer += chunk
        while buffer:
            try:
                result, index = decoder.raw_decode(buffer)
                yield result
                buffer = buffer[index:]
            except ValueError:
                # Not enough data to decode, read more
                break

def load_results(filename):
    results = []
    with open(filename, 'r') as content:
        for data in json_parse(content):
            results.append(data)
    return results

# rank_results = load_json_data('results/new_results/rank_results.json')
# cll = 1 - np.asarray(rank_results['Adversarial model'])
# majority_vote = 1 - np.asarray(rank_results['majority_vote'])
# ranks = np.arange(100) + 1

# plt.figure(figsize=[6, 2.5])
# plt.plot(ranks, cll, ':.', color="r", label='CLL')
# plt.plot(ranks, majority_vote, ':.', color="g", label='Majority Vote')
# plt.xlabel('Rank')
# plt.ylabel('Error')
# plt.xlim(xmin=0)
# # plt.ylim(ymin=0)
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.savefig('results/rank_results.pdf', bbox_inches='tight')
# plt.show()

## dependent error experiments
weak_signals = np.arange(10)
results = load_results('../results/json/cardio_error.json')[0]
all_accs = np.asarray(results['ALL'])
ge_accs = np.asarray(results['GE'])
avg_accs = np.asarray(results['AVG'])
w_avg_accs = np.asarray(results['W-AVG'])

plt.figure(figsize=[6, 3])
plt.plot(weak_signals, all_accs, color="r", label='ALL')
plt.plot(weak_signals, ge_accs, color="orange", label='GE')
plt.plot(weak_signals, avg_accs, color="g", label='AVG')
plt.plot(weak_signals, w_avg_accs, color="b", label='W-AVG')
plt.xlabel('Number of copies of WS-bad')
plt.ylabel('Accuracy')

lgd = plt.legend(loc='upper left', prop={'size': 10}, bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('../results/dependent_results.pdf',
            bbox_extra_artists=(lgd, ),
            bbox_inches='tight')
plt.show()

# weak_signals = np.arange(10) + 1

# results = load_json_data('results/all_good_signals.json')
# all_accs = np.asarray(results['ALL'])
# mv_accs = np.asarray(results['MV'])
# mmce_accs = np.asarray(results['MMCE'])

# plt.figure(figsize=[4, 2.5])
# plt.plot(weak_signals, all_accs, ':.', color="b", label='ALL')
# plt.plot(weak_signals, mmce_accs, ':.', color="r", label='MMCE')
# plt.plot(weak_signals, mv_accs, ':.', color="g", label='Majority Vote')
# plt.xlabel('Crowd labels')
# plt.ylabel('Accuracy')
# # plt.ylim(ymin=0)
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig('results/good_signal_results.pdf', bbox_inches='tight')
# plt.show()


# for plotting the bound experiments
# x1 = np.linspace(0,0.5,20)
# # y1 = [0.41035, 0.41085, 0.45161666666666667, 0.5771833333333334, 0.6126666666666667, 0.63545, 0.6369833333333333, 0.6243, 0.6149666666666667, 0.6001166666666666, 0.58285]
# y1 = load_json_data('results/bounds_accuracy.json')

# x2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# y2 = [0.733113674, 0.728720483, 0.733662823, 0.728171334, 0.732564525, 0.509610104, 0.270730368, 0.266886326, 0.264140582, 0.275672707]

# plt.figure(figsize=[6, 2.5])
# plt.scatter(x1, y1, color="C0", marker='x', label='F-MNIST (multiclass)')
# plt.xlabel('Error rates')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('results/multi_bounds.pdf', bbox_inches='tight')
# plt.show()

# plt.figure(figsize=[6, 2.5])
# plt.scatter(x2, y2, color="C1", marker='+', label='SST-2 (binary)')
# plt.xlabel('Error rates')
# plt.ylabel('Accuracy')
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('results/bounds.pdf', bbox_inches='tight')
# plt.show()

# plot text results
# data = load_json_data('results/text/imbd_unirand.json')
# true_results = load_json_data('results/text/imbd_true.json')
# min_results = load_json_data('results/text/imbd_min.json')#['Adversarial model']['cll_label_accuracy']
# max_results = load_json_data('results/text/imbd_max.json')
# bounds = load_json_data('results/text/imbd_true_bounds.json')
# weak_signals = np.arange(np.ravel(bounds['error']).size) + 1

# plt.figure(figsize=[6, 2.5])
# adversarial_model = data['Adversarial model']
# weak_model = data['Weak model']
# cll_accuracy = adversarial_model['cll_train_accuracy']
# baseline_accuracy = weak_model['baseline_train_accuracy']
# min_accuracy = min_results['Adversarial model']['cll_train_accuracy']
# max_accuracy = max_results['Adversarial model']['cll_train_accuracy']
# true_accuracy = true_results['Adversarial model']['cll_train_accuracy']
# weak_signals = np.arange(len(cll_accuracy)) + 1

# plt.plot(weak_signals, true_accuracy, 'o-', color="C0", label='True bounds')
# plt.plot(weak_signals, cll_accuracy, '^-', color="C1", label='CLL')
# plt.plot(weak_signals, baseline_accuracy, 's-', color="C2", label='Majority Vote')
# plt.plot(weak_signals, min_accuracy, ':.', color="C3", label='Min')
# plt.plot(weak_signals, max_accuracy, ':.', color="C4", label='Max')

# plt.plot(weak_signals, np.ravel(bounds['error']), ':.', color="C0", label='Error bounds')
# plt.plot(weak_signals, np.ravel(bounds['precision']), ':.', color="C1", label='Precision bounds')
# plt.xlabel('No. of weak signals')
# plt.ylabel('Error and Precision')

# lgd = plt.legend(loc='upper left', prop={'size': 10}, bbox_to_anchor=(1, 1))

# plt.xlim(xmin=0.5)
# plt.title("True bounds of the weak signals")

# plt.tight_layout()

# plt.savefig('results/text/bounds.pdf',
#             bbox_extra_artists=(lgd, ),
#             bbox_inches='tight')

# plt.show()


# for plotting histograms
# avg_train_error = []
# ntype = '_label_'

# path = 'results/'
# directories = ['random_cll', 'uniform_rand', 'min_max']

# for name in directories:
#     # for num_weak_signals in range(start, max_weak_signals + 1):
#     # filename = path+name+'/'+str(
#     #     num_weak_signals) + '_signal_results.json'
#     filename = path+name+'.json'
#     with open(filename) as json_file:
#         data = json.load(json_file)

#     adversarial_model = data['Adversarial model']

#     if name == 'random_cll':
#         weak_model = data['Weak model']
#         baseline_error = 1 - np.asarray(weak_model['baseline'+ntype+'accuracy'])
#     if name == 'uniform_rand':
#         cll = 1 - np.asarray(adversarial_model['cll'+ntype+'accuracy'])
#     if name == 'min_max':
#         min_max = 1 - np.asarray(adversarial_model['cll'+ntype+'accuracy'])

# plt.figure(figsize=[6, 2.5])
# plt.hist(cll, bins=20)
# plt.axvline(x=min_max[1],  linestyle='--', color='C0', label='Min')
# plt.axvline(x=min_max[0],  linestyle='--', color='C1', label='Max')
# plt.axvline(x=baseline_error,  linestyle='--', color='C2', label='AVG')
# plt.xlabel('Label Error')
# lgd = plt.legend(loc='upper left', prop={'size': 10}, bbox_to_anchor=(1, 1))
# plt.savefig(path+'label_errors.pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight')
# plt.show()

# get results for snorkel
# snorkel_results = load_results(path+'snorkel_results.json')
# for result in snorkel_results:
#     snorkel_label_error.append(result['snorkel_label_error'])
#     snorkel_test_error.append(1 - result['Snorkel_test_accuracy'])


# path = 'results/mscoco/'
# max_weak_signals = 5
# eps = 0.3

# cll = load_json_data(path+'mscoco_labels.json')
# snorkel = load_json_data(path+'mscoco_snorkel_labels.json')

# # constant values
# all_pe_error = 1 - np.asarray(cll['accuracy'])
# all_e_error = 1 - np.asarray(cll['error_accuracy'])
# snorkel_error = 1 - np.asarray(snorkel['accuracies'])
# mv_error = 1 - np.asarray(snorkel['majority_vote'])

# weak_signals = np.arange(1, max_weak_signals + 1)

# plt.figure(figsize=[6, 3.5])

# plt.plot(weak_signals, mv_error, '^-', color="C1", label='Majority_vote')
# plt.plot(weak_signals, snorkel_error, 's-', color="C2", label='Snorkel')
# plt.plot(weak_signals, all_pe_error, 'o-', color="C3", label='Stoch-GALL')
# plt.plot(weak_signals, all_e_error, 'o-', color="C4", label='ALL')
# plt.text(1, 0.34, '0.71', ha='center', color="C1", va='center')
# plt.text(2, 0.34, '0.78', ha='center', color="C1", va='center')
# plt.xlabel('No. of weak signals')
# plt.ylabel('Error of train labels')
# lgd = plt.legend(loc='upper left', prop={'size': 10}, bbox_to_anchor=(1, 1))
# plt.xlim(xmin=0.5)
# plt.ylim(ymax=0.345)
# plt.tight_layout()
# plt.savefig(path+'mscoco_label_accuracy.pdf',
#             bbox_extra_artists=(lgd, ),
#             bbox_inches='tight')
# plt.show()

# path = 'results/svhn/'

# semi_bounds = load_json_data(path+'convnet_model_signals.json')
# bounds = load_json_data(path+'bounds.json')

# semi_precision = np.asarray(semi_bounds['precision']).ravel()
# semi_error_rates = np.asarray(semi_bounds['error_rates']).ravel()

# precision = np.asarray(bounds['precision']).ravel()
# error_rates = np.asarray(bounds['error_rates']).ravel()

# precision = np.concatenate((semi_precision, precision))
# error_rates = np.concatenate((semi_error_rates, error_rates))

# plt.figure(figsize=[6, 2.5])
# plt.hist(error_rates, bins=20)
# plt.savefig(path+'error_bounds.pdf',bbox_inches='tight')
# plt.show()

# plt.figure(figsize=[6, 2.5])
# plt.hist(precision, bins=20)
# plt.savefig(path+'precision_bounds.pdf', bbox_inches='tight')
# plt.show()
