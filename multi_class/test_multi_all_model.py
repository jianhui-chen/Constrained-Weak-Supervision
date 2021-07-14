from multi_all_model import MultiALL
from data_utilities import read_text_data
from text_utilities import get_textsupervision_data
from setup_model import accuracy_score

all_data = read_text_data('../datasets/imdb/')

data = get_textsupervision_data(all_data, supervision='manual', true_bounds=False)
weak_model = data['weak_model']
weak_signal_probabilities = weak_model['weak_signals']
weak_signal_precision = weak_model['precision']
weak_signal_error_bounds = weak_model['error_bounds']
active_signals = weak_model['active_mask']

model_names = data['model_names']
train_data, train_labels = data['train_data']
test_data, test_labels = data['test_data']
has_labels, has_test_data = data['has_labels']

model = MultiALL()


model.fit(train_data, weak_signal_probabilities, weak_signal_error_bounds, weak_signal_precision, active_signals)

y1 = model.predict_proba(train_data)
train_acc = accuracy_score(train_labels, y1)

y2 = model.predict_proba(test_data)
test_acc = accuracy_score(test_labels, y2)


print('Stoch-gall train_acc: %f, test_accu: %f' %(train_acc, test_acc))
