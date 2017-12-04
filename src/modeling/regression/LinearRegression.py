import math
import argparse
import logging
import numpy as np
import os
import sqlite3
import sys

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import timezone
from math import sqrt
from random import randrange
from sqlite3 import Error

# notes
# https://statisticalhorizons.com/logistic-regression-for-rare-events
# ~5 *events* per predictor
# number of events is important ("effective sample size")

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
# TODO Below is dirty and probably not how things should be
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling
from modeling.regression.regression_model import RegressionModel

np.seterr(divide='ignore', invalid='ignore')        

class LinearModel(RegressionModel):
    def __init__(self):
        # TODO This simply calls super constructor and might need (or not) updates
        super().__init__()

    def regression_on_repository(self, repository_record):
        # TODO Implement this
        pass

    def serialize_to_file(self, path_to_model):
        # TODO Implement this
        pass

    def deserialize_from_file(self, path_to_model):
        # TODO Implement this
        pass

class LinearModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)
        # TODO This simply calls super constructor and might need (or not) updates

    def run_modeling(cross_validation_params, model_output_path):
        # TODO Implement this
        return LinearModel()
        
    
# Calculate root mean squared error
def rmsemet(actual, predicted):
    sumerror = 0.0
    sums = 0.0
    logging.info("Calculating root mean squared error")
    for x in range(len(actual)):
        prd_error = predicted[x] - actual[x]
        actuals = actual[x]
        sumerror += (prd_error ** 2)
        sums += (actuals**2)
    mean_error = sumerror / float(len(actual))
    mean_err = sums / float(len(actual))
    return np.sqrt(mean_error), np.sqrt(mean_err)

# Evaluate an algorithm using a train/test split
def evaluate(x, y, algorithm, *args):
    predicted = algorithm(x, y, *args)
    actual = [y[-1] for row in y]
    rmse = rmsemet(actual, predicted)
    print(predicted)
    return rmse

# calculates the logistic function
def mean(values):
    print('type = ', type(values))
    print('len = ', len(values))
    return sum((v) for v in values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += ((x[i]) - mean_x) * ((y[i]) - mean_y)
    return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
    result = []
    for j in range(len(mean)):
        s = 0
        for i in range(len(values)):
            s += (values[i, j] - mean[j]) ** 2
        if s == 0:
            print(j)
        result += [ s ]
    return result
    #return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(x):
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    print ("Coeffiecients ", b1)
    return (b0, b1)

#regression algorithm
def linear_regression(x, y, *args):
    pred = list()
    b0, b1 = coefficients(x)
    for yi in y:
        slope = b0 + b1 * yi
        pred.append(slope)
    print ("Predictions: ", pred[0])
    return pred

# determines if a repo will be active in half a year (dependent variable)
def future(con, current_time):
    succeed = 'repo_succeed_{}'.format(current_time)
    con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            repositoryID AS id,
            sum(star_count) + sum(code_push_count) + sum(fork_count) + sum(org_activity_count) AS activities
        FROM selection
        WHERE julianday(timestamp) BETWEEN julianday(?) - 91 AND julianday(?)
        GROUP BY repositoryID
    """.format(succeed), (current_time, current_time))
    activities = con.execute('SELECT activities FROM \'{}\'ORDER BY id'.format(succeed)).fetchall()
    return [ x for (x,) in activities ]

# gathers all independen variables
def ndata(con, current_time):
    # compute the average values
    avg_name = 'repo_averages_{}'.format(current_time)
    delta_name = 'repo_deltas_{}'.format(current_time)
    avgerages = con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            repositoryID as id,
            avg(star_count) as avg_star,
            avg(code_push_count) as avg_push,
            avg(pull_request_reviewed_count) as avg_pr_reviewed,
            avg(pull_request_resolved_count) as avg_pr_resolved,
            avg(fork_count) as avg_fork,
            avg(release_count) as avg_release,
            avg(issue_created_count) as avg_issue_created,
            avg(issue_commented_count) as avg_issue_commented,
            avg(issue_resolved_count) as avg_issue_resolved
        FROM selection
        WHERE timestamp <= ?
        GROUP BY repositoryID
        """.format(avg_name), (current_time,)
        )
    # compute the average change in all the attributes for this repository over the last two month (62 days)
    deltas = con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            new.repositoryID as id,
            avg(new.star_count - old.star_count) as delta_star,
            avg(new.code_push_count - old.code_push_count) as delta_push,
            avg(new.pull_request_reviewed_count - old.pull_request_reviewed_count) as delta_pr_reviewed,
            avg(new.pull_request_resolved_count - old.pull_request_resolved_count) as delta_pr_resolved,
            avg(new.fork_count - old.fork_count) as delta_fork,
            avg(new.release_count - old.release_count) as delta_release,
            avg(new.issue_created_count - old.issue_created_count) as delta_issue_created,
            avg(new.issue_commented_count - old.issue_commented_count) as delta_issue_commented,
            avg(new.issue_resolved_count - old.issue_resolved_count) as delta_issue_resolved
        FROM selection new JOIN repository old ON old.repositoryID = new.repositoryID
        WHERE julianday(new.timestamp) = julianday(old.timestamp) + 7
          AND julianday(old.timestamp) >= julianday(?) - 62
        GROUP BY new.repositoryID
            """.format(delta_name), (current_time,))
    result = con.execute("""
        SELECT
            avg_star,
            avg_push,
            avg_pr_reviewed,
            avg_pr_resolved,
            avg_fork,
            avg_release,
            avg_issue_created,
            avg_issue_commented,
            avg_issue_resolved,
            delta_star,
            delta_push,
            delta_pr_reviewed,
            delta_pr_resolved,
            delta_fork,
            delta_release,
            delta_issue_created,
            delta_issue_commented,
            delta_issue_resolved
        FROM \"{}\" avg JOIN \"{}\" delta ON avg.id = delta.id
        ORDER BY avg.id
    """.format(avg_name, delta_name)).fetchall()
    return result

if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a l regression model of GitHub repositories.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    parser.add_argument("-c", "--cache", help="Path to cache database.")

    args = parser.parse_args()

    print("Starting parameter building")

    con = sqlite3.connect(args.dataset)
    con.execute("ATTACH \"{}\" AS CACHE".format(args.cache))

    con.execute("""
            CREATE TEMP VIEW selection AS
            SELECT repository.*
            FROM repository JOIN repo on repositoryID = id
            WHERE julianday(creation_date) <= julianday('{}') - 62
            """.format('2017-03-27'))
            
    data = np.array(ndata(con, "2017-03-27"))
    success = np.array(future(con, "2017-03-27"))
    

    assert(len(data) == len(success))
    p = np.random.permutation(len(data))
    data = data[p]
    success = success[p]

    training_cutoff = math.ceil(len(data) * 0.9) # 90% train, rest test
    x = np.array(data[:training_cutoff])
    y = np.array(success[:training_cutoff])
    print("Done building parameters, starting modeling")
   
    model = linear_regression(x, y)
    
    #print(model)
    print("Done modeling, starting test")
    
    x = np.array(data[training_cutoff:])
    y = np.array(success[training_cutoff:])
    
    rsme = evaluate(x, y, linear_regression)
    print("RSME: ", (rsme))
    #lr = LinearModeling(args.dataset)
    #logging.info("Running the modeling..")
    #lr.run_modeling({})
    