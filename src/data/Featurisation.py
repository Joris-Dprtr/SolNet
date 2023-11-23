import numpy as np
import pickle


def _load_data(file_path):
    """
    load data from a given file path
    :param file_path: the file path name
    :return: return the data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: {} doesn't exist.".format(file_path))

    return data


class Featurisation:

    def __init__(self, data):
        """
        Include features for the training of the models
        :param data: the original dataframe (in list format as provided by datafetcher.py or as a file name)
        """
        if data is None:
            raise ValueError("Data cannot be None. Please provide a (list of) pandas dataframe(s) or a file path.")
        elif isinstance(data, list):
            self.data = data
        elif isinstance(data, str):
            self.data = _load_data(data)
        else:
            raise ValueError("Invalid data type provided. Must be a (list of) pandas dataframe(s) or a file path.")

    def base_features(self, featurelist):
        """
        Features to include that are already provided by PVGIS
        :param featurelist: list of features that we want to include in the model
        :return: the data list but with only the features provided in the featurelist included
        """
        for i in range(len(self.data)):
            self.data[i] = self.data[i][featurelist]

        return self.data

    def cyclic_features(self, yearly=True, daily=True):
        """
        Cyclical features to include in the model
        :param yearly: yearly cyclical features, transforming the months of the year in sin and cos features
        :param daily: daily cyclical features, transforming the hours of the day in sin and cos features
        :return: the data list but with chosen cyclic features included
        """
        for i in range(len(self.data)):
            if daily is True:
                self.data[i]['hour_sin'] = np.sin(2 * np.pi * self.data[i].index.hour / 24)
                self.data[i]['hour_cos'] = np.cos(2 * np.pi * self.data[i].index.hour / 24)
            if yearly is True:
                self.data[i]['month_sin'] = np.sin(2 * np.pi * self.data[i].index.month / 12)
                self.data[i]['month_cos'] = np.cos(2 * np.pi * self.data[i].index.month / 12)

        return self.data
