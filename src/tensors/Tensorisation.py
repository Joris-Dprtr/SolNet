import torch


def _moving_window(tensor, timesteps, prediction_length):
    """
    Create a moving window based on the steps that we want to forecast and the length of the forecast period
    :param tensor: the flat tensor
    :param timesteps: the N timesteps that we want to move forward before forecasting a new period
    :param prediction_length: the length of the period to forecast
    :return: a tensor with dimensions taking the moving window into account
    """
    length = int((len(tensor) - timesteps) / prediction_length) + 1
    moving_window = torch.zeros(length, timesteps, 1)

    for i in range(length):
        moving_window[i, :, 0] = tensor[i * prediction_length:timesteps + i * prediction_length].flatten()

    return moving_window


def _scale(train, test, domain_min=None, domain_max=None):
    """
    MinMax scaling, fitting and transforming the train set, transforming the test set (with the train set fit)
    :param train: the train tensor
    :param test: the test tensor
    :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
    :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
    :return: returns the scaled train and test sets
    """
    minimum = domain_min if domain_min is not None else train.min()
    maximum = domain_max if domain_max is not None else train.max()

    denominator = maximum - minimum or 1e-8

    train = (train - minimum) / denominator
    test = (test - minimum) / denominator

    return train, test


class Tensorisation:

    def __init__(
            self,
            data,
            target: str,
            features: list,
            lags: int,
            forecast_period: int,
            train_test_split=0.8,
            domain_min=None,
            domain_max=None):
        """
        create tensors for use in pytorch, based on the data list we get from the datafetcher.py script.
        :param data: the data (list of dataframes)
        :param target: the target variable name
        :param features: A list of feature names
        :param lags: The number of lags to include (the input length)
        :param forecast_period: the number of hours to forecast
        :param train_test_split: the train test split as a float (0.8 means 80% train data and 20% test data)
        :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
        :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
        """
        self.data = data
        self.target = target
        self.features = features
        self.lags = lags
        self.forecast_period = forecast_period
        self.train_test_split = train_test_split
        self.domain_min = domain_min
        self.domain_max = domain_max

    def tensor_creation(self):
        """
        The method doing the tensor creation
        :return: tensors, split in a train and test set, with features (X) and targets (y)
        """
        prediction_len = len(self.data) - self.lags  # See how much data is used for predictions

        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows = int(
            prediction_len / self.forecast_period)  # Get the number of predictions we can make.

        train_len = round(windows * self.train_test_split)  # Split the features into a train set...
        test_len = windows - train_len  # ... and a test set

        X_train = torch.zeros(train_len, self.lags,
                              len(self.features))  # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags,
                             len(self.features))  # Create the empty testing tensor for the features

        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):
            X_tensor = torch.tensor(self.data[feature]).type(torch.float32)
            flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
            X_train_feature = X_tensor[:flat_train_len]  # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len:-self.forecast_period]  # ... and a test set

            # Use the scaling method to get everything between 0 and 1     
            train, test = _scale(X_train_feature,
                                 X_test_feature,
                                 domain_min=self.domain_min[i] if isinstance(self.domain_max, list) else None,
                                 domain_max=self.domain_max[i] if isinstance(self.domain_max, list) else None)

            # Use the moving window to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:, :, i] = _moving_window(train,
                                              self.lags,
                                              self.forecast_period).squeeze(-1)
            X_test[:, :, i] = _moving_window(test,
                                             self.lags,
                                             self.forecast_period).squeeze(-1)

            # Make the target vector if the feature is our target
            if feature == self.target:
                y_tensor = X_tensor[self.lags:]
                y_train, y_test = _scale(y_tensor[:train_len * self.forecast_period],
                                         y_tensor[train_len * self.forecast_period:],
                                         domain_min=self.domain_min[i] if isinstance(self.domain_max,
                                                                                     list) else None,
                                         domain_max=self.domain_max[i] if isinstance(self.domain_max,
                                                                                     list) else None)

                y_train = y_train.view(train_len, self.forecast_period, 1)
                y_test = y_test.view(test_len, self.forecast_period, 1)

        return X_train, X_test, y_train, y_test

    def tensor_creation_with_evaluation(self, evaluation_length: int):
        """
        A similar method that takes into account a separate evaluation set for the purpose of transfer learning
        :param evaluation_length: the length of the evaluation set
        :return: tensors, split in a train, test and evaluation set, with features (X) and targets (y)
        """
        prediction_len = len(self.data) - self.lags - evaluation_length  # See how much data is used for predictions

        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows = int(prediction_len / self.forecast_period)  # Get the number of predictions we can make.

        train_len = round(windows * self.train_test_split)  # Split the features into a train set...
        test_len = windows - train_len  # ... and a test set
        evaluation_len = int((evaluation_length - self.lags) / self.forecast_period)

        X_train = torch.zeros(train_len, self.lags,
                              len(self.features))  # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags,
                             len(self.features))  # Create the empty testing tensor for the features
        X_eval = torch.zeros(evaluation_len, self.lags, len(self.features))

        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):
            X_tensor = torch.tensor(self.data[feature][:-evaluation_length]).type(torch.float32)
            X_tensor_eval = torch.tensor(self.data[feature][-evaluation_length:]).type(torch.float32)
            flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
            X_train_feature = X_tensor[:flat_train_len]  # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len:-self.forecast_period]  # ... and a test set
            X_eval_feature = X_tensor_eval[:-self.forecast_period]

            # Use the scaling method to get everything between 0 and 1     
            train, test = _scale(X_train_feature,
                                 X_test_feature,
                                 domain_min=self.domain_min[i] if isinstance(self.domain_max, list) else None,
                                 domain_max=self.domain_max[i] if isinstance(self.domain_max, list) else None)

            _, eval = _scale(X_train_feature,
                             X_eval_feature,
                             domain_min=self.domain_min[i] if isinstance(self.domain_max, list) else None,
                             domain_max=self.domain_max[i] if isinstance(self.domain_max, list) else None)

            # Use the moving window to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:, :, i] = _moving_window(train,
                                              self.lags,
                                              self.forecast_period).squeeze(-1)
            X_test[:, :, i] = _moving_window(test,
                                             self.lags,
                                             self.forecast_period).squeeze(-1)
            X_eval[:, :, i] = _moving_window(eval,
                                             self.lags,
                                             self.forecast_period).squeeze(-1)

            # Make the target vector if the feature is our target
            if feature == self.target:
                y_tensor = X_tensor[self.lags:]
                y_tensor_eval = X_tensor_eval[self.lags:]

                y_train, y_test = _scale(y_tensor[:train_len * self.forecast_period],
                                         y_tensor[train_len * self.forecast_period:],
                                         domain_min=self.domain_min[i] if isinstance(self.domain_max,
                                                                                     list) else None,
                                         domain_max=self.domain_max[i] if isinstance(self.domain_max,
                                                                                     list) else None)

                _, y_eval = _scale(y_tensor[:train_len * self.forecast_period],
                                   y_tensor_eval,
                                   domain_min=self.domain_min[i] if isinstance(self.domain_max, list) else None,
                                   domain_max=self.domain_max[i] if isinstance(self.domain_max, list) else None)

                y_train = y_train.view(train_len, self.forecast_period, 1)
                y_test = y_test.view(test_len, self.forecast_period, 1)
                y_eval = y_eval.view(evaluation_len, self.forecast_period, 1)

        return X_train, X_test, y_train, y_test, X_eval, y_eval
