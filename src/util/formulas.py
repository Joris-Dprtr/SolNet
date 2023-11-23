import math
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity


# Functions used for data gathering

def km_to_lat(km):
    lat = 1 / 110.574 * km
    return lat


def km_to_long(km, lat):
    long = 1 / (111.32 * abs(math.cos(math.radians(lat)))) * km
    return long


def circle_pattern(points):
    points_df = pd.DataFrame({'Points': np.linspace(0, points, points + 1)})
    points_df['Pi'] = 2 * math.pi * points_df['Points'] / points_df['Points'].max()
    points_df['Sine'] = np.sin(points_df['Pi'])
    points_df['Cosine'] = np.cos(points_df['Pi'])
    points_df = points_df[:-1]
    return points_df


# Helper functions used for evaluation

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


def interval_calc(x, num_intervals=11):
    maxi = np.max(x)
    mini = np.min(x)

    interval = (maxi - mini) / num_intervals

    maximum = round_down(maxi + interval, -len(str(int(np.modf(maxi)[1]))))
    minimum = round_up(mini - interval, -len(str(int(np.modf(mini)[1]))))

    i = 1
    while maximum < maxi or (maximum - interval) > maxi:
        maximum = round_down(maxi + interval, -(len(str(int(np.modf(maxi)[1]))) - i))
        i += i

    i = 1
    while minimum > mini or (minimum + interval) < mini:
        minimum = round_up(mini - interval, -(len(str(int(np.modf(mini)[1]))) - i))
        i += i

    intervals = np.linspace(minimum, maximum, num_intervals)

    return intervals


def x_given_y_intervals(x, y, intervals):
    interval_df = pd.DataFrame(columns=["x given y", "y interval"])

    for i in range(len(intervals) - 2):
        bools = np.vectorize(lambda x: intervals[i] <= x < intervals[i + 1])(y)
        y_interval = str("[" + "{:.1f}".format(intervals[i]) + " : " + "{:.1f}".format(intervals[i + 1]) + "]")
        locations = np.where(bools)
        for j in range(len(locations[0])):
            interval_df.loc[len(interval_df)] = [x[locations[0][j]], y_interval]

    bools = np.vectorize(lambda x: intervals[-2] <= x < intervals[-1])(y)
    y_interval = str("[" + "{:.1f}".format(intervals[-2]) + " : " + "{:.1f}".format(intervals[-1]) + "]")
    locations = np.where(bools)
    for k in range(len(locations[0])):
        interval_df.loc[len(interval_df)] = [x[locations[0][k]], y_interval]

    return interval_df


def cKDE(x, y):
    x_unique = np.unique(x)
    x_range = np.sort(x_unique)

    xy = np.vstack([x, y]).T
    bandwidth = 10
    kde_xy = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(xy)

    kde_y_given_x = []
    for xi in x_range:
        kde_xi = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1, 1))
        log_dens_xi = kde_xi.score_samples([[xi]])
        log_dens_xy = kde_xy.score_samples(np.hstack([np.full(len(x), xi).reshape(-1, 1), y.reshape(-1, 1)]))
        log_dens_y_given_xi = log_dens_xy - log_dens_xi
        kde_y_given_x.append(np.exp(log_dens_y_given_xi))

    return kde_y_given_x, x_range


# Evaluation metrics

def rmse(x, y):
    rmse = np.sqrt(np.mean(np.square(np.subtract(x, y))))
    return rmse


def var(x):
    var = np.sum(np.square(np.subtract(x, np.mean(x)))) / (len(x) - 1)
    return var


def unconditional_bias(x, y):
    unconditional_bias = np.square(np.subtract(np.mean(y), np.mean(x)))
    return unconditional_bias


def conditional_bias_1(x, y):
    y_unique = np.unique(y)
    x_conditional = []

    # TO CHANGE: REPLACE BY KCDE
    for i in range(len(y_unique)):
        bools = np.equal(y, y_unique[i])
        locations = np.where(bools)
        xGiveny = []
        for i in range(len(locations[0])):
            xGiveny.append(x[locations[0][i]])
        conditional_mean = np.mean(xGiveny)
        x_conditional.append(conditional_mean)

    mapping_dict = {y_unique[i]: x_conditional[i] for i in range(len(y_unique))}
    x_mapped = np.vectorize(mapping_dict.__getitem__)(y)

    conditional_bias_1 = np.mean(np.square(np.subtract(y, x_mapped)))

    return conditional_bias_1


def conditional_bias_1_kde(x, y, decimals=3):
    x = np.round(x, decimals)
    # y = np.round(y,decimals)

    kde_y_given_x, x_range = cKDE(x, y)

    y_given_x = np.trapz(y.reshape(-1, 1) * np.array(kde_y_given_x).T, y, axis=0) / np.trapz(np.array(kde_y_given_x).T,
                                                                                             y, axis=0)
    mapping = dict(zip(x_range, y_given_x))
    y_given_x_mapped = np.vectorize(mapping.get)(x)

    conditional_bias_1 = np.mean(np.square(np.subtract(x, y_given_x_mapped)))

    return conditional_bias_1


def resolution(x, y):
    y_unique = np.unique(y)
    x_conditional = []

    # TO CHANGE: REPLACE BY KCDE
    for i in range(len(y_unique)):
        bools = np.equal(y, y_unique[i])
        locations = np.where(bools)
        xGiveny = []
        for i in range(len(locations[0])):
            xGiveny.append(x[locations[0][i]])
        conditional_mean = np.mean(xGiveny)
        x_conditional.append(conditional_mean)

    mapping_dict = {y_unique[i]: x_conditional[i] for i in range(len(y_unique))}
    x_mapped = np.vectorize(mapping_dict.__getitem__)(y)

    resolution = np.mean(np.square(np.subtract(x_mapped, np.mean(x))))

    return resolution


def conditional_bias_2(x, y):
    x_unique = np.unique(x)
    y_conditional = []

    # TO CHANGE: REPLACE BY KCDE
    for i in range(len(x_unique)):
        bools = np.equal(x, x_unique[i])
        locations = np.where(bools)
        yGivenx = []
        for i in range(len(locations[0])):
            yGivenx.append(y[locations[0][i]])
        conditional_mean = np.mean(yGivenx)
        y_conditional.append(conditional_mean)

    mapping_dict = {x_unique[i]: y_conditional[i] for i in range(len(x_unique))}
    y_mapped = np.vectorize(mapping_dict.__getitem__)(x)

    conditional_bias_2 = np.mean(np.square(np.subtract(x, y_mapped)))

    return conditional_bias_2


def discrimination(x, y):
    x_unique = np.unique(x)
    y_conditional = []

    # TO CHANGE: REPLACE BY KCDE
    for i in range(len(x_unique)):
        bools = np.equal(x, x_unique[i])
        locations = np.where(bools)
        yGivenx = []
        for i in range(len(locations[0])):
            yGivenx.append(x[locations[0][i]])
        conditional_mean = np.mean(yGivenx)
        y_conditional.append(conditional_mean)

    mapping_dict = {x_unique[i]: y_conditional[i] for i in range(len(x_unique))}
    y_mapped = np.vectorize(mapping_dict.__getitem__)(x)

    discrimination = np.mean(np.square(np.subtract(y_mapped, np.mean(y))))

    return discrimination
