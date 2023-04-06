import torch
    
def remove_nighttime(self):

    dataset_sum = self.data[0]

    for i in range(len(self.data)-1):
        dataset_sum = dataset_sum.add(self.data[i+1])

    hourly_output = dataset_sum['P'].groupby(dataset_sum.index.hour).sum()
    hours_with_zero_output = hourly_output.loc[hourly_output == 0].index.tolist()
    hours_with_non_zero_output = hourly_output.loc[hourly_output != 0].index.tolist()
    
    for i in range(len(self.data)):
        self.data[i] = self.data[i][~self.data[i].index.hour.isin(hours_with_zero_output)]
    
    return self.data, hours_with_zero_output, hours_with_non_zero_output


def get_full_days(tensors, hours_with_non_zero_output):
    full_days = torch.empty(0)

    for i in range(len(tensors)):
        day = torch.zeros(24)
        day[hours_with_non_zero_output] = tensors[i].squeeze()
        full_days = torch.cat((full_days, day))

    return full_days
