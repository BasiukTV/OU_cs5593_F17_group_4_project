import os, sys
import math

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

class LogisticModel(RegressionModel):
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

class LogisticModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)
        # TODO This simply calls super constructor and might need (or not) updates

    def run_modeling(self, cross_validation_params):
        # TODO Implement this
        return LogisticModel()
import numpy as np
from scipy.sparse import csr_matrix
import math
import sqlite3

THRESH = 0.0000001

# calculates the logistic function
def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def linear_combination(x, a):
    t = a[0] # intercept
    for i in range(len(x)):
        t += x[i] * a[i + 1]
    return t

def predict(x, weights):
    # t is a linear combination of the expanatory variables:
    # t = w0 + x1 * w1 + x2 * w2 + ... + xn * wn
    t = (linear_combination(x, weights))
    return sigmoid(t)

# likelihood of the given observations occurring with given probabilities (binomial model, assumes independent)
# this is the function that we (indirectly, through the log) want to maximize
def likelihood(observations, probabilities):
    l = 1
    for (i, yi) in enumerate(observations):
        if yi == 0:
            l *= (1 - probabilities[i])
        elif yi == 1:
            l *= probabilities[i]
        else:
            raise("Observations have to be binary")
    return l

# odds = p / (1-p) (occuring/not_occuring)
# odds of y = 1 given x (and the weights)
# this equals e^logit(...)
# this is *not* a probability, but ranges from 0 to +infty
def odds(x, weights):
    math.exp(linear_combination(x, weights))

# inverse of logistic function
def logit(t):
    return math.log(t / (1 - t))

# test the inverting (should return 3, with a bit of error)
# print(logit(sigmoid(3)))

# test on study example using wikipedias weights
# for hours in range(1,6):
#     print(round(predict([hours], [-4.0777, 1.5046]), 2))

# x is input data, y in {0, 1} is data to predict, P is probability vector
def log_likelihood_gradient(x, y, P):
    a = y * x - P * x
    return 42

def probability(observations, weight):
    s = observations.shape
    num_observations = s[0]
    num_variables = s[1]
    P = np.zeros(num_observations)
    exponent = np.zeros(num_observations)

    for i in range(0, num_observations):
        for j in range(0, num_variables):
            exponent[i] += observations[i, j] * weight[j]
        P[i] = math.exp(exponent[i]) / (1 + math.exp(exponent[i]))
    return P

def generateB(P):
    import gc
    from scipy.sparse import diags
    gc.collect()
    num_observations = np.shape(P)[0]
    B_diag = np.zeros(num_observations)
    for i in range(0, num_observations):
        B_diag[i] = P[i] * (1 - P[i])
    B = diags(B_diag)

    return B

def logistic_regression(x, y):
    s = np.shape(x)
    num_observations = s[0]
    num_variables = s[1]
    assert(np.shape(y) == (num_observations,))

    weight = np.zeros(num_variables + 1)
    bias = np.ones((num_observations, 1))

    design_matrix = np.append(bias, x, 1)
    transposed_design = np.transpose(design_matrix)
    # from scipy.sparse import csr_matrix
    # design_matrix = csr_matrix(design_matrix)
    # transposed_design = csr_matrix(transposed_design)

    change = np.ones(num_variables + 1)

    diff = 1
    while diff > THRESH:
        P = probability(design_matrix, weight)
        B = generateB(P)
        likelihood_gradient = transposed_design.dot(P - y)
        likelihood_hessian = transposed_design.dot(B.dot(design_matrix))
        hessian_inv_approx = np.linalg.lstsq(likelihood_hessian, np.eye(num_variables + 1, num_variables + 1))[0]
        change = - hessian_inv_approx.dot(likelihood_gradient)
        weight = weight + change
        diff = 0
        for val in change:
            diff += val ** 2
        print(weight)
        # TODO calculatell

    return weight

# x = np.transpose(np.array(range(1,11), ndmin=2))
# y = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
# model = logistic_regression(x, y)
# print(predict(x, model))

# x = np.array(
#         [[0.5, 1], [0.75, 1], [1.0, 1], [1.25, 1], [1.5, 1], [1.75, 1], [2.0, 1], [2.25, 1], [2.5, 1], [2.75, 1], [3.0, 1], [3.25, 1], [4.0, 1], [4.25, 1], [4.5, 1], [4.75, 1], [5.0, 1], [5.5, 1]])
# y = np.array(
#     [0,   0,    0,   0,    0,   1,    0,   1,    0,   1,    0,   1,    1,   1,    1,   1,    1,   1])
# model = logistic_regression(x, y)
# print(model)
# print(predict(x, model))

# determines if a repo will be active in half a year (dependent variable)
def look_into_future(con, id, current_time):
    # pushes over 1 month in half a year
    pushes = con.execute("""
        SELECT sum(code_push_count) + sum(pull_request_resolved_count) + sum(issue_resolved_count) + sum(org_activity_count)
        FROM repository
        WHERE repositoryID = ?
          AND julianday(timestamp) BETWEEN julianday(?) + 91 AND julianday(?) + 182
    """, (id, current_time, current_time)).fetchone()[0]
    return 1 if pushes > 0 else 0

# gathers all independen variables
def gather_repo_data(con, id, current_time):
    # compute the average values
    averages = con.execute("""
        SELECT
            avg(star_count),
            avg(total_contributor_count),
            avg(contributor_type1_count),
            avg(contributor_type2_count),
            avg(contributor_type3_count),
            avg(contributor_type4_count),
            avg(contributor_type5_count),
            avg(code_push_count),
            avg(pull_request_created_count),
            avg(pull_request_reviewed_count),
            avg(pull_request_resolved_count),
            avg(fork_count),
            avg(release_count),
            avg(issue_created_count),
            avg(issue_commented_count),
            avg(issue_resolved_count),
            avg(org_activity_count)
        FROM repository
        WHERE repositoryID = ?
          AND timestamp <= ?
        """, (id, current_time))
    # compute the average change in all the attributes for this repository over the last two month (62 days)
    deltas = con.execute("""
        SELECT
            avg(new.star_count - old.star_count),
            avg(new.total_contributor_count - old.total_contributor_count),
            avg(new.contributor_type1_count - old.contributor_type1_count),
            avg(new.contributor_type2_count - old.contributor_type2_count),
            avg(new.contributor_type3_count - old.contributor_type3_count),
            avg(new.contributor_type4_count - old.contributor_type4_count),
            avg(new.contributor_type5_count - old.contributor_type5_count),
            avg(new.code_push_count - old.code_push_count),
            avg(new.pull_request_created_count - old.pull_request_created_count),
            avg(new.pull_request_reviewed_count - old.pull_request_reviewed_count),
            avg(new.pull_request_resolved_count - old.pull_request_resolved_count),
            avg(new.fork_count - old.fork_count),
            avg(new.release_count - old.release_count),
            avg(new.issue_created_count - old.issue_created_count),
            avg(new.issue_commented_count - old.issue_commented_count),
            avg(new.issue_resolved_count - old.issue_resolved_count),
            avg(new.org_activity_count - old.org_activity_count)
        FROM repository new, repository old
        WHERE new.repositoryID = ? and old.repositoryID = ?
          AND julianday(new.timestamp) = julianday(old.timestamp) + 7
          AND julianday(old.timestamp) >= julianday(?) - 62
            """, (id, id, current_time))
    result = averages.fetchone() + deltas.fetchone()
    return tuple(xi for xi in result if xi is not None) # remove Nones (if contributors are not known yet)


if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a logistic regression model of GitHub repositories.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    args = parser.parse_args()

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    # modeling = LogisticModeling(args.dataset)
    # model = modeling.run_modeling("not_actual_cross_validation_params")
    # model.serialize_to_file("not_an_anctual_path_to_file")

    print("Starting parameter building")

    con = sqlite3.connect(args.dataset)
    int_count = 0
    total_count = 0
    data = []
    activity = []
    # I need 182 days to be able to look into the future and 62 days to compute the delta
    for (id,) in con.execute("SELECT id FROM repo WHERE finished=1 AND julianday(creation_date) <= julianday('2017-03-27') - 244 ORDER BY RANDOM()"):
        total_count += 1
        active = look_into_future(con, id, "2017-03-27")
        if active == 1:
            int_count += 1

        result = gather_repo_data(con, id, "2017-03-27")
        data += [ result ]
        activity += [ active ]
    print("{} out of {} are events ({}%)".format(int_count, total_count, round(int_count/total_count * 100, 4)))

    training_cutoff = math.ceil(len(data) * 0.9) # 90% train, rest test
    x = np.array(data[:training_cutoff])
    y = np.array(activity[:training_cutoff])
    print("Done building parameters, starting modeling")
    import cProfile
    model = cProfile.run('logistic_regression(x, y)')
    print(model)
    print("Done modeling, starting test")
    x = np.array(data[training_cutoff:])
    y = np.array(activity[training_cutoff:])
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for (xi, yi) in zip(x, y):
        prediction = predict(xi, model)
        if yi == 1:
            if xi == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if xi == 1:
                false_positive += 1
            else:
                true_negative += 1
    print("True positive: {}".format(true_positive))
    print("False positive: {}".format(false_positive))
    print("True negative: {}".format(true_negative))
    print("False negative: {}".format(false_negative))
    print("Total accuracy: {}%".format(round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100, 2)))
    print("Event accuracy: {}%".format(round(true_positive / (true_positive + false_negative) * 100, 2)))
    print("Non-Event accuracy: {}%".format(round(true_negative / (true_negative + false_positive) * 100, 2)))
