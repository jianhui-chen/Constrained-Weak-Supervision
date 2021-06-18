
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



"""
This is an extra file to show progress on the updated version of the Data Class
Can be later used with the overhead file running all the code
"""

class Data:

    def __load_and_process_data(self, datapath, load_data):

        """
            Calls function that gets the data, and then splits it into training, validation, and testing sets

            :param datapath: location of data
            :type datapath: string
            :param load_data: funtion specific to data set that gets data from the path and cleans it
            :type load_data: function
        """

        data_matrix, data_labels = load_data(datapath)

        #Split the data into 70% training and 30% test set
        train_dev_data, test_data, train_dev_labels, test_labels = train_test_split(data_matrix, data_labels.astype('float'), test_size=0.3, shuffle=True, stratify=data_labels)

        #Normalize the features of the data
        scaler = preprocessing.StandardScaler().fit(train_dev_data)
        train_dev_data = scaler.transform(train_dev_data)
        test_data = scaler.transform(test_data)

        assert train_dev_labels.size == train_dev_data.shape[0]
        assert test_labels.size == test_data.shape[0]

        data = {}

        #Split the remaining data into 57.15% training and 42.85% test set
        val_data, weak_supervision_data, val_labels, weak_supervision_labels = train_test_split(train_dev_data, train_dev_labels.astype('float'), test_size=0.4285, shuffle=True, stratify=train_dev_labels)

        data['dev_data'] = weak_supervision_data, weak_supervision_labels
        data['validation_data'] = val_data, val_labels
        data['test_data'] = test_data, test_labels

        return data
    
    def __init__(self, name, views, datapath, savepath, load_data):
        self.n = name
        self.v = views
        self.dp = datapath
        self.sp = savepath
        self.data = self.__load_and_process_data(datapath, load_data)
    


    

    