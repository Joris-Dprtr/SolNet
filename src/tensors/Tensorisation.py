import torch

class Tensorisation():
    
    def __init__(
        self, 
        data, 
        target, 
        features, 
        lags, 
        forecast_period, 
        train_test_split = 0.8):
         
         self.data = data,
         self.target = target,
         self.features = features,
         self.lags = lags,
         self.forecast_period = forecast_period
         self.train_test_split = train_test_split

    def _moving_window(self, tensor, timesteps, prediction_length):
        length = int((len(tensor)-timesteps) / prediction_length) + 1
        moving_window = torch.zeros(length, timesteps, 1)

        for i in range(length):
            moving_window[i,:,0] = tensor[i*prediction_length:timesteps+i*prediction_length].flatten()

        return moving_window

    def _scale(self, train, test):
            
            minimum = train.min().item()
            maximum = train.max().item()

            train = (train - minimum) / (maximum - minimum)
            test = (test - minimum) / (maximum - minimum)

            return train, test

    def tensor_creation(self):

        y_len = len(self.data) - self.lags
        extra_data = y_len%self.forecast_period
        y_len -= extra_data
        predictions  = y_len/self.forecast_period

        train_len = int(predictions * self.train_test_split)
        test_len = int(predictions - train_len)

        X_train = torch.zeros(train_len, self.lags, len(self.features))
        X_test = torch.zeros(test_len, self.lags, len(self.features))

        for i, feature in enumerate(self.features):
            feature_tensor = torch.tensor(self.data[feature][:len(self.data) - extra_data]).type(torch.float32)
            X_tensor = feature_tensor[:-self.forecast_period]
            X_train_feature = X_tensor[:self.lags + train_len * self.forecast_period - self.forecast_period]
            X_test_feature = X_tensor[train_len * self.forecast_period:]
            train, test = self._scale(X_train_feature, X_test_feature)
            X_train[:,:,i] = self._moving_window(train, self.lags, self.forecast_period).squeeze(-1)
            X_test[:,:,i] = self._moving_window(test, self.lags, self.forecast_period).squeeze(-1)
            
            if(feature == self.target):
                y_tensor = feature_tensor[self.lags:]
                y_train, y_test = self._scale(y_tensor[:train_len*self.forecast_period], y_tensor[train_len*self.forecast_period:])
                y_train = y_train.view(train_len, self.forecast_period, 1)
                y_test = y_test.view(test_len, self.forecast_period, 1)

        return X_train, X_test, y_train, y_test
