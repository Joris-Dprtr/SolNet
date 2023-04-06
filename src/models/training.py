import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


class training():

    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs,
        batch_size = 32,
        learning_rate = 0.001,
        criterion = torch.nn.MSELoss() 
        ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        self.train_loader = DataLoader(train_data, batch_size = batch_size)
        self.test_loader = DataLoader(test_data, batch_size = batch_size)

        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def fit(self):
        
        avg_train_error = []
        avg_test_error = []
        state_dict_list = []

        for epoch in range(self.epochs):
            num_train_batches = 0
            num_test_batches = 0
            total_loss = 0
            total_loss_test = 0
            batches = iter(self.train_loader)
            self.model.train()
            
            for input, output in batches:
                
                prediction = self.model(input)
                output = output.squeeze()
                loss = self.criterion(prediction,output)
                
                total_loss += float(loss)
                num_train_batches += 1
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            
            with torch.inference_mode():
            
                test_batches = iter(self.test_loader)
            
                for input, output in test_batches:
                    prediction = self.model(input)
                    output = output.squeeze()
                    test_loss = self.criterion(prediction,output)

                    total_loss_test += float(test_loss)
                    num_test_batches += 1

            avg_train_error.append(total_loss/num_train_batches)
            avg_test_error.append(total_loss_test/num_test_batches)

            state_dict_list.append(self.model.state_dict())
            
            if(epoch%5 == 0):
                print('Step {}: Average train loss: {:.4f} | Average test loss: {:.4f}'.format((epoch), avg_train_error[epoch], avg_test_error[epoch]))
                
        argmin_test = avg_test_error.index(min(avg_test_error))        
                
        print('Best Epoch: ' + str(argmin_test))

        plt.plot(avg_train_error,label='train error')
        plt.plot(avg_test_error, label='test error')
        plt.legend()
        
        return state_dict_list, argmin_test

        
    def load_best_model(self, state_dict_list, argmin_test):
        self.model.load_state_dict(state_dict_list[argmin_test])
 
        
    def save_model(self, name):
        torch.save(self.model.state_dict(), '../../models/' + str(name))