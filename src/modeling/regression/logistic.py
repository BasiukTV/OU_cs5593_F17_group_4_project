import os, sys
import math
import numpy as np
import math
import sqlite3
from scipy.sparse import csr_matrix
import sympy

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

THRESH = 0.0000001
PARAM_NAMES = [
    'avg_star',
    'avg_push',
    'avg_pr_created',
    'avg_pr_reviewed',
    'avg_pr_resolved',
    'avg_fork',
    'avg_release',
    'avg_issue_created',
    'avg_issue_commented',
    'avg_issue_resolved',
    'avg_org',
    'delta_star',
    'delta_push',
    'delta_pr_created',
    'delta_pr_reviewed',
    # 'delta_pr_resolved',
    # 'delta_fork',
    # 'delta_release',
    'delta_issue_created',
    'delta_issue_commented',
    'delta_issue_resolved',
    'delta_org'
]

# calculates the logistic function
def sigmoid(t):
    return 1 / (1 + sympy.exp(-t))

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
    sympy.exp(linear_combination(x, weights))

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
        a = sympy.exp(exponent[i])
        P[i] = a / (1 + a)
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
    r = 0
    while diff > THRESH:
        r += 1
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
        print("Round {}".format(r))
        for (n, w) in zip(PARAM_NAMES, weight):
            print("{:28}{}".format(n, round(w, 2)))
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

# determines if a repo is active (dependent variable)
def judge_active(con, current_time):
    active_name = "repo_active_{}".format(current_time)
    # acitivities over 3 month
    con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            repositoryID AS id,
            sum(code_push_count) + sum(pull_request_resolved_count) + sum(issue_resolved_count) + sum(org_activity_count) AS activities
        FROM selection
        WHERE julianday(timestamp) BETWEEN julianday(?) - 91 AND julianday(?)
        GROUP BY repositoryID
    """.format(active_name), (current_time, current_time))
    activities = con.execute("SELECT activities FROM \"{}\" ORDER BY id".format(active_name)).fetchall()
    return [ x for (x,) in activities ]

# gathers all independen variables
def gather_repo_data(con, current_time):
    # compute the average values
    avg_name = 'repo_averages_{}'.format(current_time)
    delta_name = 'repo_deltas_{}'.format(current_time)
    con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            repositoryID as id,
            avg(star_count) as avg_star,
            avg(total_contributor_count) as avg_contrib,
            avg(contributor_type1_count) as avg_contrib_1,
            avg(contributor_type2_count) as avg_contrib_2,
            avg(contributor_type3_count) as avg_contrib_3,
            avg(contributor_type4_count) as avg_contrib_4,
            avg(contributor_type5_count) as avg_contrib_5,
            avg(code_push_count) as avg_push,
            avg(pull_request_created_count) as avg_pr_created,
            avg(pull_request_reviewed_count) as avg_pr_reviewed,
            avg(pull_request_resolved_count) as avg_pr_resolved,
            avg(fork_count) as avg_fork,
            avg(release_count) as avg_release,
            avg(issue_created_count) as avg_issue_created,
            avg(issue_commented_count) as avg_issue_commented,
            avg(issue_resolved_count) as avg_issue_resolved,
            avg(org_activity_count) as avg_org
        FROM selection
        WHERE timestamp <= ?
        GROUP BY repositoryID
        """.format(avg_name), (current_time,))
    # compute the average change in all the attributes for this repository over the last two month (62 days)
    con.execute("""
        CREATE TABLE IF NOT EXISTS CACHE.\"{}\" AS
        SELECT
            new.repositoryID as id,
            avg(new.star_count - old.star_count) as delta_star,
            avg(new.total_contributor_count - old.total_contributor_count) as delta_contrib,
            avg(new.contributor_type1_count - old.contributor_type1_count) as delta_contrib_1,
            avg(new.contributor_type2_count - old.contributor_type2_count) as delta_contrib_2,
            avg(new.contributor_type3_count - old.contributor_type3_count) as delta_contrib_3,
            avg(new.contributor_type4_count - old.contributor_type4_count) as delta_contrib_4,
            avg(new.contributor_type5_count - old.contributor_type5_count) as delta_contrib_5,
            avg(new.code_push_count - old.code_push_count) as delta_push,
            avg(new.pull_request_created_count - old.pull_request_created_count) as delta_pr_created,
            avg(new.pull_request_reviewed_count - old.pull_request_reviewed_count) as delta_pr_reviewed,
            avg(new.pull_request_resolved_count - old.pull_request_resolved_count) as delta_pr_resolved,
            avg(new.fork_count - old.fork_count) as delta_fork,
            avg(new.release_count - old.release_count) as delta_release,
            avg(new.issue_created_count - old.issue_created_count) as delta_issue_created,
            avg(new.issue_commented_count - old.issue_commented_count) as delta_issue_commented,
            avg(new.issue_resolved_count - old.issue_resolved_count) as delta_issue_resolved,
            avg(new.org_activity_count - old.org_activity_count) as delta_org
        FROM selection new JOIN selection old ON old.repositoryID = new.repositoryID
        WHERE julianday(new.timestamp) = julianday(old.timestamp) + 7
          AND julianday(old.timestamp) BETWEEN julianday(?) - 62 AND julianday(?)
        GROUP BY new.repositoryID
            """.format(delta_name), (current_time, current_time))
    averages = con.execute("""
        SELECT
            avg_star,
            avg_push,
            avg_pr_created,
            avg_pr_reviewed,
            avg_pr_resolved,
            avg_fork,
            avg_release,
            avg_issue_created,
            avg_issue_commented,
            avg_issue_resolved,
            avg_org
        FROM \"{}\"
        ORDER BY id
    """.format(avg_name)).fetchall()
    deltas = con.execute("""
        SELECT
            delta_star,
            delta_push,
            delta_pr_created,
            delta_pr_reviewed,
            delta_issue_created,
            delta_issue_commented,
            delta_issue_resolved,
            delta_org
        FROM \"{}\"
        ORDER BY id
    """.format(delta_name)).fetchall()
    return np.append(averages, deltas, 1)


if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a logistic regression model of GitHub repositories.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    parser.add_argument("-c", "--cache", help="Path to cache database.")
    args = parser.parse_args()

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    # modeling = LogisticModeling(args.dataset)
    # model = modeling.run_modeling("not_actual_cross_validation_params")
    # model.serialize_to_file("not_an_anctual_path_to_file")

    print("Starting parameter building")

    con = sqlite3.connect(args.dataset)
    con.execute("ATTACH \"{}\" AS CACHE".format(args.cache))
    int_count = 0
    total_count = 0


    con.execute("""
        CREATE TEMP VIEW selection AS
        SELECT repository.*
        FROM repository JOIN repo on repositoryID = id
        WHERE julianday(creation_date) <= julianday('{}') - 62
    """.format('2017-03-27'))
    data = np.array(gather_repo_data(con, "2017-03-27"))
    activities = np.array(judge_active(con, "2017-09-27"))

    # randomize both arrays while keeping the indices together
    assert(len(data) == len(activities))
    p = np.random.permutation(len(data))
    data = data[p]
    activities = activities[p]

    training_cutoff = math.ceil(len(data) * 0.9) # 90% train, rest test
    x = data[:training_cutoff]
    y = activities[:training_cutoff]
    print("Done building parameters, starting modeling")
    # import cProfile
    # model = cProfile.run('logistic_regression(x, y)')
    model = logistic_regression(x, y)
    print(model)
    print("Done modeling, starting test")
    x = np.array(data[training_cutoff:])
    y = np.array(activities[training_cutoff:])
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for (xi, yi) in zip(x, y):
        prediction = predict(xi, model)
        if yi == 1:
            if prediction == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if prediction == 1:
                false_positive += 1
            else:
                true_negative += 1
    print("True positive:  {}".format(true_positive))
    print("False positive: {}".format(false_positive))
    print("True negative:  {}".format(true_negative))
    print("False negative: {}".format(false_negative))
    print("Total accuracy: {}%".format(0 if (true_positive + true_negative == 0) else round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100, 2)))
    print("Event accuracy: {}%".format(0 if (true_positive == 0) else round(true_positive / (true_positive + false_negative) * 100, 2)))
    print("Non-Event accuracy: {}%".format(0 if (true_negative == 0) else round(true_negative / (true_negative + false_positive) * 100, 2)))
