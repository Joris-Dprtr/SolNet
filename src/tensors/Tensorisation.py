import torch

class Tensorisation():
    
    def __init__(
        self, 
        data, 
        target, 
        features, 
        lags: int, 
        forecast_period: int, 
        train_test_split = 0.8,
        domain_min = None,
        domain_max = None):
         
        self.data = data
        self.target = target
        self.features = features
        self.lags = lags
        self.forecast_period = forecast_period
        self.train_test_split = train_test_split
        self.domain_min = domain_min
        self.domain_max = domain_max
        
    def _moving_window(self, tensor, timesteps, prediction_length):
        length = int((len(tensor)-timesteps) / prediction_length) + 1
        moving_window = torch.zeros(length, timesteps, 1)

        for i in range(length):
            moving_window[i,:,0] = tensor[i*prediction_length:timesteps+i*prediction_length].flatten()

        return moving_window


    def _scale(self, train, test, domain_min = None, domain_max = None):

        minimum = domain_min if domain_min is not None else train.min()
        maximum = domain_max if domain_max is not None else train.max()
            
        denominator = maximum - minimum or 1e-8

        train = (train - minimum) / denominator
        test = (test - minimum) / denominator

        return train, test


    def tensor_creation(self):
        
        prediction_len = len(self.data) - self.lags                                             # See how much data is used for predictions
        
        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows  = int(prediction_len/self.forecast_period)                                     # Get the number of predictions we can make. This is the actual length of the first dimension of our data.

        train_len = round(windows * self.train_test_split)                                      # Split the features into a train set...
        test_len = windows - train_len                                                          # ... and a test set
  
        X_train = torch.zeros(train_len, self.lags, len(self.features))                         # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags, len(self.features))                           # Create the empty testing tensor for the features 

        flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period  # We need this when we split the, still flattened, feature from the dataframe in a train and test set

        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):                                            
            X_tensor = torch.tensor(self.data[feature]).type(torch.float32)
            flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
            X_train_feature = X_tensor[:flat_train_len]                                                             # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len:-self.forecast_period]                                         # ... and a test set (we re move the last prediction length since it is not used)
            
            # Use the scaling method to get everything between 0 and 1     
            train, test = self._scale(X_train_feature, 
                                      X_test_feature, 
                                      domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                      domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)

            # Use the moving window method to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:,:,i] = self._moving_window(train, 
                                                 self.lags, 
                                                 self.forecast_period).squeeze(-1)
            X_test[:,:,i] = self._moving_window(test, 
                                                self.lags, 
                                                self.forecast_period).squeeze(-1)
            
            # Make the target vector if the feature is our target
            if(feature == self.target):
                y_tensor = X_tensor[self.lags:]                                                 
                y_train, y_test = self._scale(y_tensor[:train_len*self.forecast_period], 
                                              y_tensor[train_len*self.forecast_period:], 
                                              domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                              domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)

                y_train = y_train.view(train_len, self.forecast_period, 1)
                y_test = y_test.view(test_len, self.forecast_period, 1)

        return X_train, X_test, y_train, y_test


    def tensor_creation_with_evaluation(self, evaluation_length):
        
        prediction_len = len(self.data) - self.lags - evaluation_length                         # See how much data is used for predictions
        
        # The number of windows we have to predict depends on the length of the forecast window 
        # (we assume that the forecaster wants to forecast every upcoming period)
        windows  = int(prediction_len/self.forecast_period)                                     # Get the number of predictions we can make. This is the actual length of the first dimension of our data.
        
        train_len = round(windows * self.train_test_split)                                      # Split the features into a train set...
        test_len = windows - train_len                                                          # ... and a test set
        evaluation_len = int((evaluation_length-self.lags)/self.forecast_period)

        X_train = torch.zeros(train_len, self.lags, len(self.features))                         # Create the empty training tensor for the features
        X_test = torch.zeros(test_len, self.lags, len(self.features))                           # Create the empty testing tensor for the features 
        X_eval = torch.zeros(evaluation_len, self.lags, len(self.features))

        flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period  # We need this when we split the, still flattened, feature from the dataframe in a train and test set

        # Iterate over all the features to populate the empty tensors
        for i, feature in enumerate(self.features):                                            
            X_tensor = torch.tensor(self.data[feature][:-evaluation_length]).type(torch.float32)
            X_tensor_eval = torch.tensor(self.data[feature][-evaluation_length:]).type(torch.float32)
            flat_train_len = (train_len * self.forecast_period) + self.lags - self.forecast_period
            X_train_feature = X_tensor[:flat_train_len]                                                             # Split the flattened dataframe in a train set...
            X_test_feature = X_tensor[flat_train_len:-self.forecast_period]                                         # ... and a test set (we re move the last prediction length since it is not used)
            X_eval_feature = X_tensor_eval[:-self.forecast_period]
            
            # Use the scaling method to get everything between 0 and 1     
            train, test = self._scale(X_train_feature, 
                                      X_test_feature, 
                                      domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                      domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)

            _, eval = self._scale(X_train_feature, 
                                      X_eval_feature, 
                                      domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                      domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)
            
            # Use the moving window method to go from the flat tensor to the correct dimensions (window, lags per window)
            X_train[:,:,i] = self._moving_window(train, 
                                                 self.lags, 
                                                 self.forecast_period).squeeze(-1)
            X_test[:,:,i] = self._moving_window(test, 
                                                self.lags, 
                                                self.forecast_period).squeeze(-1)
            X_eval[:,:,i] = self._moving_window(eval, 
                                                self.lags, 
                                                self.forecast_period).squeeze(-1)
            
            # Make the target vector if the feature is our target
            if(feature == self.target):
                y_tensor = X_tensor[self.lags:]         
                y_tensor_eval = X_tensor_eval[self.lags:]                                        

                y_train, y_test = self._scale(y_tensor[:train_len*self.forecast_period], 
                                              y_tensor[train_len*self.forecast_period:], 
                                              domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                              domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)
                
                _, y_eval = self._scale(y_tensor[:train_len*self.forecast_period], 
                                              y_tensor_eval, 
                                              domain_min=self.domain_min[i] if isinstance(self.domain_max,list) else None, 
                                              domain_max=self.domain_max[i] if isinstance(self.domain_max,list) else None)
                
                y_train = y_train.view(train_len, self.forecast_period, 1)
                y_test = y_test.view(test_len, self.forecast_period, 1)
                y_eval = y_eval.view(evaluation_len, self.forecast_period, 1)

        return X_train, X_test, y_train, y_test, X_eval, y_eval