import math
import numpy as np
import pandas as pd

def kmToLat(km):
    lat = 1/110.574 * km
    return lat

def kmToLong(km, lat):
    long = 1/(111.32*abs(math.cos(math.radians(lat)))) * km
    return long

def circle_pattern(points):
    points_df = pd.DataFrame({'Points':np.linspace(0,points,points+1)})
    points_df['Pi'] = 2 * math.pi * points_df['Points'] / points_df['Points'].max()
    points_df['Sine'] = np.sin(points_df['Pi'])
    points_df['Cosine'] = np.cos(points_df['Pi'])
    points_df = points_df[:-1]
    return points_df

def mse(x,y):
    mse = np.mean(np.square(np.subtract(x,y)))
    return mse

def var(x):
    var = np.sum(np.square(np.subtract(x,np.mean(x)))) / (len(x)-1)
    return var

def unconditional_bias(x,y):
    unconditional_bias = np.square(np.subtract(np.mean(y),np.mean(x)))
    return unconditional_bias

def conditional_bias_1(x,y):
    
    y_unique = np.unique(y)
    x_conditional = []
    
    for i in range(len(y_unique)):
        bools = np.equal(y,y_unique[i])
        locations = np.where(bools)
        xGiveny = []
        for i in range(len(locations[0])):
            xGiveny.append(x[locations[0][i]]) 
        conditional_mean = np.mean(xGiveny)
        x_conditional.append(conditional_mean)
    
    conditional_bias_1 = np.mean(np.square(np.subtract(y_unique,x_conditional)))

    return conditional_bias_1

def resolution(x,y):
    
    y_unique = np.unique(y)
    x_conditional = []
    
    for i in range(len(y_unique)):
        bools = np.equal(y,y_unique[i])
        locations = np.where(bools)
        xGiveny = []
        for i in range(len(locations[0])):
            xGiveny.append(x[locations[0][i]]) 
        conditional_mean = np.mean(xGiveny)
        x_conditional.append(conditional_mean)
    
    resolution = np.mean(np.square(np.subtract(x_conditional,np.mean(x))))

    return resolution

def conditional_bias_2(x,y):
    
    x_unique = np.unique(x)
    y_conditional = []
    
    for i in range(len(x_unique)):
        bools = np.equal(x,x_unique[i])
        locations = np.where(bools)
        yGivenx = []
        for i in range(len(locations[0])):
            yGivenx.append(y[locations[0][i]]) 
        conditional_mean = np.mean(yGivenx)
        y_conditional.append(conditional_mean)
    
    conditional_bias_2 = np.mean(np.square(np.subtract(x_unique,y_conditional)))

    return conditional_bias_2

def discrimination(x,y):
    
    x_unique = np.unique(x)
    y_conditional = []
    
    for i in range(len(x_unique)):
        bools = np.equal(x,x_unique[i])
        locations = np.where(bools)
        yGivenx = []
        for i in range(len(locations[0])):
            yGivenx.append(x[locations[0][i]]) 
        conditional_mean = np.mean(yGivenx)
        y_conditional.append(conditional_mean)
    
    discrimination = np.mean(np.square(np.subtract(y_conditional,np.mean(y))))

    return discrimination
