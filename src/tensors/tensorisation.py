import torch
import numpy as np


def _moving_window(array, windows, window_size, window_step):
    """
    Create a moving window based on the steps that we want to forecast
    :param array: the flat array
    :param window_size: the N timesteps that we want to include in each window
    :param window_step: the step size
    :return: an array with dimensions taking the moving window into account
    """
    # Create an array of starting indices for each window
    start_indices = np.arange(0, windows * window_step, window_step)
    # Fill the array from the start index onwards
    index_array = start_indices[:, np.newaxis] + np.arange(window_size)
    # Apply the indexing on the original array
    return array[index_array]


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


class Tensors:

    def __init__(self,
                 data,
                 target: str,
                 past_features: list,
                 future_features: list,
                 lags: int,
                 forecast_period: int,
                 gap: int = 0,
                 forecast_gap: int = 0,
                 train_test_split: float = 0.8,
                 evaluation_length: int = 0,
                 domain_min=None,
                 domain_max=None
                 ):
        """
        create tensors for use in pytorch, based on the data list we get from the datafetcher.py script.
        :param data: the data (Pandas Dataframe)
        :param target: the target variable name
        :param past_features: A list of PAST feature names
        :param future_features: A list of FUTURE feature names
        :param lags: The number of lags to include (the input length)
        :param forecast_period: the number of hours to forecast
        :param gap: the gap between the lags and the forecast period
        :param forecast_gap: the gap between each forecast, a moving window which can be negative
        :param train_test_split: the train test split as a float (0.8 means 80% train data and 20% test data)
        :param domain_min: a domain minimum (if known, otherwise it's based on the train.min())
        :param domain_max: a domain maximum (if known, otherwise it's based on the train.max())
        """

        self.data = data
        self.target = target
        self.past_features = past_features
        self.future_features = future_features
        self.lags = lags
        self.forecast_period = forecast_period
        self.gap = gap
        self.forecast_gap = forecast_gap
        self.train_test_split = train_test_split
        self.flat_evaluation_length = evaluation_length
        self.domain_min = domain_min
        self.domain_max = domain_max

    def create_tensor(self):
        """
        The method doing the tensor creation
        :return: tensors, split in a train and test set, with features (X) and targets (y)
        """

        # we can't use the first [lags] + [gap] timesteps
        prediction_length = len(self.data) - self.lags - self.gap - self.flat_evaluation_length

        # The number of predictions is based on the forecast gap + forecast length
        divider = self.forecast_gap + self.forecast_period

        # Check if the array is of the correct length
        left_over = prediction_length % divider
        if left_over != 0:
            self.data = self.data[left_over:]
            prediction_length = len(self.data) - self.lags - self.gap - self.flat_evaluation_length

        predictions = (prediction_length - self.forecast_period) // divider + 1

        train_length = round(predictions * self.train_test_split)
        test_length = predictions - train_length
        evaluation_length = max(int((self.flat_evaluation_length - self.lags - self.gap) / divider), 0)

        # Features
        # Make all the tensors
        X_train = torch.zeros(train_length, max(self.forecast_period, self.lags),
                              len(self.past_features) + len(self.future_features))
        X_test = torch.zeros(test_length, max(self.forecast_period, self.lags),
                             len(self.past_features) + len(self.future_features))
        X_eval = torch.zeros(evaluation_length, max(self.forecast_period, self.lags),
                             len(self.past_features) + len(self.future_features))

        # Past features
        # Get the flat length of the train
        past_train_start = 0
        past_train_end = (train_length - 1) * divider + self.lags
        # and test array
        past_test_start = past_train_end + self.forecast_gap + self.forecast_period - self.lags
        past_test_end = -(self.flat_evaluation_length + self.forecast_period + self.gap)

        past_eval_start = past_test_end + self.forecast_gap + self.forecast_period - self.lags
        past_eval_end = -(self.forecast_period + self.forecast_period + self.gap)

        for i, feature in enumerate(self.past_features):

            past_train_array = np.array(self.data[feature][past_train_start:past_train_end])
            past_train_array_shaped = _moving_window(past_train_array, X_train.shape[0], self.lags,
                                                     self.forecast_gap + self.forecast_period)

            past_test_array = np.array(self.data[feature][past_test_start:past_test_end])
            past_test_array_shaped = _moving_window(past_test_array, X_test.shape[0], self.lags,
                                                    self.forecast_gap + self.forecast_period)

            if self.flat_evaluation_length != 0:
                past_eval_array = np.array(self.data[feature][past_eval_start:past_eval_end])
                past_eval_array_shaped = _moving_window(past_eval_array, X_eval.shape[0], self.lags,
                                                        self.forecast_gap + self.forecast_period)

            if self.lags < self.forecast_period:
                past_train_padding = np.zeros((past_train_array_shaped.shape[0], self.forecast_period - self.lags))
                past_train_array_shaped = np.concatenate((past_train_padding, past_train_array_shaped), axis=1)

                past_test_padding = np.zeros((past_test_array_shaped.shape[0], self.forecast_period - self.lags))
                past_test_array_shaped = np.concatenate((past_test_padding, past_test_array_shaped), axis=1)

                if self.flat_evaluation_length != 0:
                    past_eval_padding = np.zeros((past_eval_array_shaped.shape[0], self.forecast_period - self.lags))
                    past_eval_array_shaped = np.concatenate((past_eval_padding, past_eval_array_shaped), axis=1)

            past_train, past_test = _scale(past_train_array_shaped, past_test_array_shaped,
                                           domain_min=self.domain_min[i] if isinstance(self.domain_max,
                                                                                       list) else None,
                                           domain_max=self.domain_max[i] if isinstance(self.domain_max,
                                                                                       list) else None)
            if self.flat_evaluation_length != 0:
                _, past_eval = _scale(past_train_array_shaped, past_eval_array_shaped,
                                      domain_min=self.domain_min[i] if isinstance(self.domain_max,
                                                                                  list) else None,
                                      domain_max=self.domain_max[i] if isinstance(self.domain_max,
                                                                                  list) else None)

            X_train[:, :, i] = torch.tensor(past_train).type(torch.float32)
            X_test[:, :, i] = torch.tensor(past_test).type(torch.float32)
            if self.flat_evaluation_length != 0:
                X_eval[:, :, i] = torch.tensor(past_eval).type(torch.float32)

        # Future features
        # Get the flat length of the train
        future_train_start = self.lags + self.gap
        future_train_end = future_train_start + (train_length - 1) * divider + self.forecast_period
        # test array
        future_test_start = future_train_end + self.forecast_gap
        future_test_end = None if (self.flat_evaluation_length == 0) else -self.flat_evaluation_length
        # and evaluation array
        future_eval_start = None if (self.flat_evaluation_length == 0) else future_test_end + self.forecast_gap
        future_eval_end = None

        for i, feature in enumerate(self.future_features):

            future_train_array = np.array(self.data[feature][future_train_start:future_train_end])
            future_train_array_shaped = _moving_window(future_train_array, X_train.shape[0], self.forecast_period,
                                                       self.forecast_gap + self.forecast_period)

            future_test_array = np.array(self.data[feature][future_test_start:future_test_end])
            future_test_array_shaped = _moving_window(future_test_array, X_test.shape[0], self.forecast_period,
                                                      self.forecast_gap + self.forecast_period)

            if self.flat_evaluation_length != 0:
                future_eval_array = np.array(self.data[feature][future_eval_start:future_eval_end])
                future_eval_array_shaped = _moving_window(future_eval_array, X_eval.shape[0], self.forecast_period,
                                                          self.forecast_gap + self.forecast_period)

            if self.lags > self.forecast_period:
                future_train_padding = np.zeros((future_train_array_shaped.shape[0], self.lags - self.forecast_period))
                future_train_array_shaped = np.concatenate((future_train_padding, future_train_array_shaped), axis=1)

                future_test_padding = np.zeros((future_test_array_shaped.shape[0], self.lags - self.forecast_period))
                future_test_array_shaped = np.concatenate((future_test_padding, future_test_array_shaped), axis=1)

                if self.flat_evaluation_length != 0:
                    future_eval_padding = np.zeros((future_eval_array_shaped.shape[0], self.lags - self.forecast_period))
                    future_eval_array_shaped = np.concatenate((future_eval_padding, future_eval_array_shaped), axis=1)

            future_train, future_test = _scale(future_train_array_shaped, future_test_array_shaped,
                                               domain_min=self.domain_min[len(self.past_features)+i] if
                                               isinstance(self.domain_max, list) else None,
                                               domain_max=self.domain_max[len(self.past_features)+i] if
                                               isinstance(self.domain_max, list) else None)

            if self.flat_evaluation_length != 0:
                _, future_eval = _scale(future_train_array_shaped, future_eval_array_shaped,
                                        domain_min=self.domain_min[len(self.past_features)+i] if
                                        isinstance(self.domain_max, list) else None,
                                        domain_max=self.domain_max[len(self.past_features)+i] if
                                        isinstance(self.domain_max, list) else None)

            X_train[:, :, len(self.past_features)+i] = torch.tensor(future_train).type(torch.float32)
            X_test[:, :, len(self.past_features)+i] = torch.tensor(future_test).type(torch.float32)

            if self.flat_evaluation_length != 0:
                X_eval[:, :, len(self.past_features)+i] = torch.tensor(future_eval).type(torch.float32)

        # Target
        # train
        target_train_array = np.array(self.data[self.target][future_train_start:future_train_end])
        target_train_array_shaped = _moving_window(target_train_array, X_train.shape[0], self.forecast_period,
                                                   self.forecast_gap + self.forecast_period)

        # test
        target_test_array = np.array(self.data[self.target][future_test_start:future_test_end])
        target_test_array_shaped = _moving_window(target_test_array, X_test.shape[0], self.forecast_period,
                                                  self.forecast_gap + self.forecast_period)

        target_train, target_test = _scale(target_train_array_shaped, target_test_array_shaped,
                                           domain_min=self.domain_min[0] if
                                           isinstance(self.domain_max, list) else None,
                                           domain_max=self.domain_max[0] if
                                           isinstance(self.domain_max, list) else None)

        y_train = torch.tensor(target_train).type(torch.float32)
        y_test = torch.tensor(target_test).type(torch.float32)

        if self.flat_evaluation_length != 0:
            target_eval_array = np.array(self.data[self.target][future_eval_start:])
            target_eval_array_shaped = _moving_window(target_eval_array, X_eval.shape[0], self.forecast_period,
                                                      self.forecast_gap + self.forecast_period)

            _, target_eval = _scale(target_train_array_shaped, target_eval_array_shaped,
                                    domain_min=self.domain_min[0] if
                                    isinstance(self.domain_max, list) else None,
                                    domain_max=self.domain_max[0] if
                                    isinstance(self.domain_max, list) else None)

            y_eval = torch.tensor(target_eval).type(torch.float32)

            return X_train, X_test, X_eval, y_train, y_test, y_eval

        else:
            return X_train, X_test, y_train, y_test
