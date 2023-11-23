import torch


def remove_nighttime(data):
    """
    Remove the nighttime hours from a dataset (hours with 0 output throughout the dataset)
    :param data: the dataset from which to remove the nighttime values
    :return: the cleaned dataset, two lists giving the nighttime hours and the daytime hours
    """
    dataset_sum = data[0]

    for i in range(len(data) - 1):
        dataset_sum = dataset_sum.add(data[i + 1])

    hourly_output = dataset_sum['P'].groupby(dataset_sum.index.hour).sum()
    hours_with_zero_output = hourly_output.loc[hourly_output == 0].index.tolist()
    hours_with_non_zero_output = hourly_output.loc[hourly_output != 0].index.tolist()

    for i in range(len(data)):
        data[i] = data[i][~data[i].index.hour.isin(hours_with_zero_output)]

    return data, hours_with_zero_output, hours_with_non_zero_output


def get_full_days(tensors, hours_with_non_zero_output):
    """
    Reverse the remove_nighttime() method and include the nighttime hours again
    :param tensors: the tensorised data
    :param hours_with_non_zero_output: the list holding the daytime hours
    :return: tensors that include the zero output (nighttime) hours
    """
    full_days = torch.empty(0)

    for i in range(len(tensors)):
        day = torch.zeros(24)
        day[hours_with_non_zero_output] = tensors[i].squeeze()
        full_days = torch.cat((full_days, day))

    return full_days
