
######################################
# FUNCTIONS HELPFUL FOR DATA SCIENCE #
######################################

import re 
import math
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *
from scipy.stats import binom, beta, norm, t, poisson, ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Declare 'x' and 'y' to SymPy
x,y = symbols('x', 'y')

# Convert a date column in a pandas dataframe to unix time
def convert_to_unix_time(df, column_name, date_format):
    df[column_name] = pd.to_datetime(df[column_name], format=date_format)
    df[column_name] = df[column_name].apply(lambda x: int(x.timestamp()))
    
# Move a column_name to be the last column in the dataframe
def move_column_to_last(df, column_name):
    # 1. Get the list of column names
    columns = df.columns.tolist()
    # 2. Remove the column
    columns.remove(column_name)
    # 3. Append the removed column name back to the list
    columns.append(column_name)
    # 4. Reorder the DataFrame
    df = df[columns]

# Count the number of missing values (NA/null) values for a dataframe column
def missing_count_in_column(x):
    return sum(x.isnull())

# Count the number of missing values (NA/null) values for a dataframe of all columns
def missing_count_all_column(df):
    return df.apply(missing_count_in_column, axis = 0)

# Count the number of missing values (NA/null) values for a dataframe of all rows
def missing_count_all_row(df):
    return df.apply(missing_count_in_column, axis = 1)

# Returns an array of all emails found in a given text
def extract_emails_text(text):
    re.findall(r"([\w.-]+@[\w.-]+)", text)

# Remove all emojis from a given text
def remove_emojis_text(text):
    text.encode('ascii', 'ignore').decode('ascii')

# Returns a new dataframe with only the categorical variables as columns
def get_catagorical_vars(df):
    return df.select_dtypes("object")

# Returns a new dataframe with only the numerical variables as columns
def get_catagorical_vars(df):
    return df.select_dtypes("number")

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

# Flip fair coin n times
def flip_coin(n):
    if n <= 40:
        raise ValueError("n should be larger than 40")
    # outcomes can be visualized as a distribution in a histogram
    outcomes = []
    outcomes_prob = []
    probabilities = np.array([0.5, 0.5])
    # Normalize the probabilities so they sum to 1, because of precision rounding errors in computer calculations
    probabilities /= probabilities.sum()
    for i in range(10000):
        flips = np.random.choice(['heads', 'tails'], size=n, p=probabilities)
        num_heads = np.sum(flips == 'heads')
        num_heads_prob = (num_heads / n)
        outcomes.append(num_heads)
        outcomes_prob.append(num_heads_prob)
    confidence_95 = np.percentile(outcomes, [2.5,97.5])
    confidence_95_prob = np.percentile(outcomes_prob, [2.5,97.5])
    return {
        "distribution": outcomes,
        "confidence_95": confidence_95,
        "distribution_prob": outcomes_prob,
        "confidence_95_prob": confidence_95_prob
    }

# Calculate the odds from a probabilty
def get_odds(probability):
    probability = float(probability)
    return probability / (1.0 - probability)

# Calculate the probability from odds
def get_probability_from_odds(odds):
    return float(odds / (1.0 + odds))

# Get the probability of a single event occurring
def get_probability_event(event, all_possible_outcomes):
    return float(event / len(all_possible_outcomes))

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

# Creates a binomial distribution using the probability mass function
def get_binomial_distribution(probability, trials):
    distribution = []
    for k in range(trials + 1):
        prob = binom.pmf(k, trials, probability)
        distribution.append(prob)
    return distribution

# Get the probability for a specific value in number of trials with a given probability per trial using the PMF
def get_pmf_value(value, probability, trials):
    return binom.pmf(value, n=trials, p=probability)

# Returns the probabality of event x or less occuring
# for a given sample_size and probability
def get_cdf_less_than(event_x, sample_size: int, probability: float):
    return binom.cdf(event_x, sample_size, probability)

# Returns the probabality of event x or more occuring
# for a given sample_size and probability
def get_cdf_greater_than(event_x, sample_size: int, probability: float):
    return 1 - binom.cdf(event_x, sample_size, probability)

# Returns the probabality of events a to b occuring
# for a given sample_size and probability
def get_cdf_range(event_a, event_b, sample_size: int, probability: float):
    if event_a >= event_b:
        raise ValueError("event_a can not be larger or equal than event_b")
    return binom.cdf(event_b, sample_size, probability) - binom.cdf(event_a, sample_size, probability)

# Beta distribution
# Returns the probability(underlying success rate) that an event with a successes and b failures has a probability or less of occuring.
def get_beta_cdf(probability, successes, failures):
    return beta.cdf(probability, successes, failures)

# Returns the probability(underlying success rate) that an event with a successes and b failures has a between prob_a and prob_b of occurring.
def get_beta_cdf_range(prob_a, prob_b, successes, failures):
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

# Returns how many standard deviations a datapoint is from the mean.
def get_number_of_standard_deviations(datapoint, mean, std):
    difference = datapoint - mean
    return difference / std

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

# Normal Cumulative Density Function
# Returns the probability of x_point or less occurring
def normal_cdf_less_than(x_point, mean, std_dev):
    return norm.cdf(x_point, mean, std_dev)

# Returns the probability of x_point or more occurring
def normal_cdf_greater_than(x_point, mean, std_dev):
    return 1 - norm.cdf(x_point, mean, std_dev)

# Returns the probability of a range between x_point_a and x_point_b occuring
def normal_cdf_range(x_point_a, x_point_b, mean, std_dev):
    if x_point_a >= x_point_b:
        raise ValueError("x_point_a can not be larger or equal than x_point_b")
    return norm.cdf(x_point_b, mean, std_dev) - norm.cdf(x_point_a, mean, std_dev)

# Inverse CDF
# Returns that a value has a probability or less of occuring
def inverse_cdf(probability: float, mean, std_dev):
    return norm.ppf(probability, loc=mean, scale=std_dev)

# Poisson Probability Mass Function
# average = How many events on average are expected to happen for a given time or timeframe
# events = Calculate the probability of how many events could happen for a given time or timeframe
def get_poisson_pmf(events, average):
    return poisson.pmf(events, average)

# Poisson Cumulative Density Function
# average = How many events on average are expected to happen for a given time or timeframe
# events = Calculate the probability of how many events or less could happen for a given time or timeframe
def get_poisson_less_than(events, average):
    return poisson.cdf(events, average)

# average = How many events on average are expected to happen for a given time or timeframe
# events = Calculate the probability of how many events or less could happen for a given time or timeframe
def get_poisson_greater_than(events, average):
    return 1 - poisson.cdf(events, average)

# average = How many events on average are expected to happen for a given time or timeframe
# events_a, events_b = Calculate the probability of how many events_a to events_b could happen for a given time or timeframe
def get_poisson_range(events_a, events_b, average):
    if events_a >= events_a:
        raise ValueError("x_point_a can not be larger or equal than x_point_b")
    return poisson.cdf(events_b, average) - poisson.cdf(events_a, average)

# mean = How many events on average are expected to happen for a given time or timeframe
# sample_size = Number of total events happened in a given time or timeframe
def generate_random_poisson_distribution(mean, sample_size):
    return poisson.rvs(mean, size = sample_size)

# generate_random_numbers_based_on_normal_distribution
def generate_random_normal_distribution(numbers, mean, std_dev):
    distribution = []
    for i in range(0, numbers):
        random_p = random.uniform(0.0, 1.0)
        random_value = norm.ppf(random_p, loc=mean, scale=std_dev)
        distribution.append(random_value)
    return distribution

# Returns a random sample of size sample_size from the population
def create_random_sample_from_population(population, sample_size):
    return np.random.choice(np.array(population), sample_size, replace = False)

# Convert an x_point to a z-score. Take an x-value and scale it in terms of standard deviation
def get_z_score(x_point, mean, std_dev):
    return (x_point - mean) / std_dev

# Convert a z-score to an x_point
def z_score_to_x(z_score, mean, std_dev):
    return (z_score * std_dev) + mean

def standard_normal_distribution():
    return norm(loc=0.0, scale=1.0)

# n = number of trials
# Expectation
def get_expected_value_binomial(n, probability):
    return n * probability

def get_variance_binomial(n, probability):
    return n * probability * (1 - probability)

# level of confidence is a probability
def critical_z_value(level_of_confidence: float):
    if level_of_confidence <= 0.0 or level_of_confidence >= 1.0:
        raise ValueError("level_of_confidence must be between 0.0 and 1.0")
    norm_dist = standard_normal_distribution()
    left_tail_area = (1.0 - level_of_confidence) / 2.0
    upper_area = 1.0 - ((1.0 - level_of_confidence) / 2.0)
    return [norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)]

# standard error of the estimate of the mean
# As sample size increases, the standard error will decrease.
# As the population standard deviation increases, so will the standard error.
# The standard error estimates the variability across multiple samples of a population.
# The standard deviation describes variability within a single sample.
def get_standard_error_mean(sample_size, std_dev_sample):
    return std_dev_sample / math.sqrt(sample_size)

def get_margin_of_error(critical_z, sample_size, std_dev_sample):
    if sample_size < 31:
        raise ValueError("sample_size must be greater than 31")
    return critical_z * get_standard_error_mean(sample_size, std_dev_sample)

# confidence interval is a range calculation showing
# how confidently a sample mean falls in a range for the population mean.
def get_confidence_interval(level_of_confidence: float, sample_size: int, mean_sample, std_dev_sample):
    critical_z = critical_z_value(level_of_confidence)
    margin_of_error = get_margin_of_error(critical_z, sample_size, std_dev_sample)
    return [mean_sample - margin_of_error, mean_sample + margin_of_error]

# Get which value range lies in 95% of the sample
# Between 2.5 and 97.5 percent of the sample
def get_95_percentile(sample):
    return np.percentile(sample, [2.5, 97.5])

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
    
# Perform a 1 sample T-Test using scipy
# Returns the t-stat and p-value
def onesamp_ttest(distribution, expected_mean):
    tstat, pval = ttest_1samp(distribution, expected_mean)
    return [tstat, pval]
    
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
    
def encode_categorical_values_to_numbers(df, column_name):
    # Create a label encoder
    label_encoder = LabelEncoder()
    # Fit and transform the column to label-encoded values
    df[column_name] = label_encoder.fit_transform(df[column_name])
