
######################################
# FUNCTIONS HELPFUL FOR DATA SCIENCE #
######################################

import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *
from scipy.stats import binom, beta, norm, t
from sklearn.linear_model import LinearRegression

# Declare 'x' and 'y' to SymPy
x,y = symbols('x', 'y')

# f = function to calculate derivative function of
# example:
# f = x**2
def get_derivative_function(f):
    # Calculate the derivative of the function
    dx_f = diff(f)
    return dx_f

# Calculate the partial derivative functions of x and y of a function
# example:
# f = 2*x**3 + 3*y**3
def get_partial_derivative_functions(f):
    dx_f = diff(f, x)
    dy_f = diff(f, y)
    return {
        'dx_f': dx_f,
        'dy_f': dy_f
    }

# Calculate the integral of the function with respect to x
# for the area between x = point_xa and point_xb
def get_integral(f, point_xa, point_xb):
    area = integrate(f, (x, point_xa, point_xb))
    return area

# Calculate the odds from a probabilty
def get_odds(probability):
    probability = float(probability)
    return probability / (1.0 - probability)

# Calculate the probability from odds
def get_probability_from_odds(odds):
    return float(odds / (1.0 + odds))

# Calculate the probability of 2 independent events occurring.
# P1 AND P2
# p1 = probability of event 1 occurring
# p2 = probability of event 2 occurring
def get_joint_probability(p1, p2):
    return p1 * p2

# Calculate the probability of mutually exclusive events occurring
# P1 OR P2. 
def get_untion_probability_mutual(p1, p2):
    return p1 + p2 #- (get_joint_probability(p1, p2))

# Calculate the probability of non-mutually exclusive events occurring
# P1 OR P2. 
def get_untion_probability_non_mutual(p1, p2):
    return p1 + p2 - (get_joint_probability(p1, p2))

def get_binomial_distribution(probability, trials):
    distribution = []
    for k in range(trials + 1):
        prob = binom.pmf(k, trials, probability)
        distribution.append(prob)
    return distribution

# Beta distribution
# Returns the probability(underlying success rate) that an event with a successes and b failures has a probability or less of occuring.
def get_cumalative_density_function(probability, successes, failures):
    return beta.cdf(probability, successes, failures)

# Returns the probability(underlying success rate) that an event with a successes and b failures has a between prob_a and prob_b of occurring.
def get_cdf_range(prob_a, prob_b, successes, failures):
    if prob_a >= prob_b:
        raise ValueError("prob_a can not be larger or equal than prob_b")
    return beta.cdf(prob_b, successes, failures) - beta.cdf(prob_a, successes, failures)

# Returns the mean
# values is an array
def get_mean(values):
    return sum(values) / len(values)

# Returns the weighted mean
# datapoints is an array
# weights is an array of floats
def get_weighted_mean(weights, values):
    return sum(v * w for v,w in zip(values, weights))

# Returns the median
def get_median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = int(n / 2) - 1 if n % 2 == 0 else int(n/2)
    if n % 2 == 0:
        return (ordered[mid] + ordered[mid+1]) / 2.0
    else:
        return ordered[mid]

# Returns the mode(s)
def get_mode(values):
    counts = defaultdict(lambda: 0)
    for s in values:
        counts[s] += 1
    max_count = max(counts.values())
    modes = [v for v in set(values) if counts[v] == max_count]
    return modes

# Returns the variance of a population
def get_variance_population(values, is_sample: bool = False):
    mean = sum(values) / len(values)
    _variance = sum((value - mean) ** 2 for value in values) / len(values) 
    return _variance

# Returns the std of a population
def get_standard_deviation_population(values):
    return math.sqrt(get_variance_population(values))

# Returns the variance of a sample
def get_variance_sample(values):
    mean = sum(values) / len(values) 
    _variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return _variance

# Returns the std of a sample
def get_standard_deviation_sample(values):
    return math.sqrt(get_variance_sample(values))

# quantifies how spread out a distribution is
def get_coefficient_of_variation(mean, std_dev):
    return std_dev / mean

# returns likelihood at value x in the PDF
def normal_probability_density_function(x_point: float, mean: float, std_dev: float) -> float:
    return (1.0 / (2.0 * math.pi * std_dev ** 2) ** 0.5) * math.exp(-1.0 * ((x_point - mean) ** 2 / (2.0 * std_dev ** 2)))

# graph the PDF for a given range
def normal_pdf_range(mean, std_dev, start_range, end_range, step=0.1):
    distribution = []
    for i in range(start_range, end_range, step):
        distribution.append(normal_probability_density_function(i, mean, std_dev))
    return distribution

#Cumulative Density Function
# Returns the probability of x_point occurring
def normal_cdf(x_point, mean, std_dev):
    return norm.cdf(x_point, mean, std_dev)

# Returns the probability of a range between x_point_a and x_point_b occuring
def normal_cdf_range(x_point_a, x_point_b, mean, std_dev):
    if x_point_a >= x_point_b:
        raise ValueError("x_point_a can not be larger or equal than x_point_b")
    return norm.cdf(x_point_b, mean, std_dev) - norm.cdf(x_point_a, mean, std_dev)

# Inverse CDF
# Returns that a value has a probability or less of occuring
def inverse_cdf(probability: float, mean, std_dev):
    return norm.ppf(probability, loc=mean, scale=std_dev)

# generate_random_numbers_based_on_normal_distribution
def generate_random_normal_distribution(numbers, mean, std_dev):
    distribution = []
    for i in range(0, numbers):
        random_p = random.uniform(0.0, 1.0)
        random_value = norm.ppf(random_p, loc=mean, scale=std_dev)
        distribution.append(random_value)
    return distribution

# Convert an x_point to a z-score. Take an x-value and scale it in terms of standard deviation
def get_z_score(x_point, mean, std_dev):
    return (x_point - mean) / std_dev

# Convert a z-score to an x_point
def z_score_to_x(z_score, mean, std_dev):
    return (z_score * std_dev) + mean

def standard_normal_distribution():
    return norm(loc=0.0, scale=1.0)

# level of confidence is a probability
def critical_z_value(level_of_confidence: float):
    if level_of_confidence <= 0.0 or level_of_confidence >= 1.0:
        raise ValueError("level_of_confidence must be between 0.0 and 1.0")
    norm_dist = standard_normal_distribution()
    left_tail_area = (1.0 - level_of_confidence) / 2.0
    upper_area = 1.0 - ((1.0 - level_of_confidence) / 2.0)
    return [norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)]

def get_margin_of_error(critical_z, sample_size, std_dev_sample):
    if sample_size < 31:
        raise ValueError("sample_size must be greater than 31")
    return critical_z * (std_dev_sample / math.sqrt(sample_size))

# confidence interval is a range calculation showing
# how confidently a sample mean falls in a range for the population mean.
def get_confidence_interval(level_of_confidence: float, sample_size: int, mean_sample, std_dev_sample):
    critical_z = critical_z_value(level_of_confidence)
    margin_of_error = get_margin_of_error(critical_z, sample_size, std_dev_sample)
    return [mean_sample - margin_of_error, mean_sample + margin_of_error]

# For 95% confidence level
# Used for sample sizes smaller than 31
def get_critical_t_value_range(sample_size):
    if sample_size > 31:
        raise ValueError("sample_size must be less than 31")
    lower = t.ppf(.025, df=sample_size-1)
    upper = t.ppf(.975, df=sample_size-1)
    return [lower, upper]
    
# p value For 95%
def get_p_value(mean, std_dev):
    # Calculate the x-value that has 2.5% of area behind it.
    lower_bound = norm.ppf(.025, mean, std_dev)
    # Calculate the x-value that has 97.5% of area behind it.
    upper_bound = norm.ppf(.975, mean, std_dev)
    
    p1 = norm.cdf(lower_bound, mean, std_dev) # p value of lower tail
    p2 = norm.cdf(upper_bound, mean, std_dev) # p value of upper tail
    # P-value of both tails
    return {
        "range": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
        "p_values": {
                "lower_tail": p1,
                "upper_tail": p2,
                "both_tails": p1 + p2
            }
        }

# Get the m and b values for a linear regression using sklearn
def get_m_b_linear_regression(data):
    # Extract input variables (all rows, all columns but last column)
    X = data.values[:, :-1]
    # Extract output column (all rows, last column)
    Y = data.values[:, -1]
    # Fit a line to the points
    fit = LinearRegression().fit(X, Y)
    m = fit.coef_.flatten()
    b = fit.intercept_.flatten()
    return {
        'm': m,
        'b': b
    }

# Get the m and b values for a linear regression using closed form solution
# Note that this may be slow and can only work for 2D data.
def get_m_b_linear_regression_closed_form(points):
    sample_size = len(points)

# Calculate the sum of squares for a linear regression line
def get_sum_of_squares(points, m, b):
    sum_of_squares = 0.0
    # calculate sum of squares
    for p in points:
        y_actual = p.y
        y_predict = m*p.x + b
        residual_squared = (y_predict - y_actual)**2
        sum_of_squares += residual_squared
        
    return sum_of_squares
    
# Plot the linear regression line
def plot_linear_regression(data, m, b):
    # Extract input variables (all rows, all columns but last column)
    X = data.values[:, :-1]
    # Extract output column (all rows, last column)
    Y = data.values[:, -1]
    # show in chart
    plt.plot(X, Y, 'o') # scatterplot
    plt.plot(X, m*X+b) # line
    plt.show()
