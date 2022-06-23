# Weakly-Supervised-Learning
This repository contains code for the following papers
    * Adversarial Label Learning
    * A General Framework for Adversarial Label Learning
    * Constrained Labeling for Weakly Supervised Learning
    * Data Consistency for Weakly Supervised Learning


NOTE: DO WE NEED THIS????
If you use this work in an academic study, please cite our paper

```
@inproceedings{arachie2019adversarial,
  title={Adversarial label learning},
  author={Arachie, Chidubem and Huang, Bert},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3183--3190},
  year={2019}
}
```

# Requirements

The library is tested in Python 3.6 and 3.7. Its main requirements are
scipy and numpy. Scikit-learn is also required to run the experiments

NOTE: Maybe add:
    *tensorflow?


# Algorithms

old_ALL:
Is built off of the BaseClassifier abstract class in Baseclassifier.py. The most important script is the old_ALL.py script that contains implementation of the algorithm inside of the old_ALL class.

ALL (aka MultiAlL):
Is built off of the BaseClassifier abstract class in Baseclassifier.py. The most important script is the old_ALL.py script that contains implementation of the algorithm inside of the ALL class.

CLL:
Is built off of the Constraint Estimator abstract class in ConstraintEstimators.py. The most important script is the ConstraintEstimators.py script that contains implementation of the algorithm inside of the CLL class.

Data Consistancy:
Is built off of the Label Estimator abstract class in LabelEstimators.py. The most important script is the LabelEstimators.py script that contains implementation of the algorithm inside of the DataConsistency class.


# Examples

We have provided a run_experiment that runs experiments on the real datasets provided on all 4 algorithms. Running experiment on other user datasets is fairly easy to implement.

# Logging

Logging is done via TensorBoard and each run is stored by the date/time the expirment was started, and then by dataset, and then by algorithm. Use:

tensorboard --logdir=logs/data_and_time/data_set/algorithm

Example: 

tensorboard --logdir=logs/2021_07_28-05:50:52_PM/breast-cancer/CLL



# Models

NOTE: FIX THIS
old_ALL:
The model trained is a logistic regression classifier as reported in the paper. The train_classifier code can be modified to use more advanced classifier

ALL (aka MultiAlL):

CLL:

Data Consistancy:

# Bounds

NOTE: FIX THIS
old_ALL:
The model is set to use the True bounds of the data. When this bounds is unknown, the user can provide an upper bounds for the weak signals or use constant bounds in the experiments scripts

ALL (aka MultiAlL):

CLL:

Data Consistancy:


# Limitations

The old_ALL algorithm only supports binary classification and weak signals that do not abstain, code for ALL (aka Multi ALL)  fixes these limitations and returns similar results to that of old_ALL. For now, we skip over running these data sets on 
