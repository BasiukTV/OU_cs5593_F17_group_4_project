import os, sys
import math
import numpy as np
import math
import sqlite3
import sympy
import json
import io

# notes
# https://statisticalhorizons.com/logistic-regression-for-rare-events
# ~5 *events* per predictor
# number of events is important ("effective sample size")

TRAINING_FRACTION = 0.9
THRESH = 0.0000001
TRAINING_TIME = "2017-03-27"
LATEST_RECORD = "2017-09-27"

PARAM_NAMES = [
    'avg_star',
    'avg_push',
    'avg_pr_created',
    # 'avg_pr_reviewed',
    # 'avg_pr_resolved',
    # 'avg_fork',
    'avg_release',
    'avg_issue_created',
    # 'avg_issue_commented',
    # 'avg_issue_resolved',
    # 'avg_org',
    'avg_contrib',
    'avg_contrib_1',
    'avg_contrib_2',
    # 'avg_contrib_3',
    'delta_star',
    'delta_push',
    'delta_pr_created'
    # 'delta_pr_reviewed',
    # 'delta_pr_resolved',
    # 'delta_fork',
    # 'delta_release',
    # 'delta_issue_created',
    # 'delta_issue_commented',
    # 'delta_issue_resolved',
    # 'delta_org'
]

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling
from modeling.regression.regression_model import RegressionModel

class LogisticModel(RegressionModel):
    def __init__(self):
        super().__init__()

    def regression_on_repository(self, repository_record):
        # t is a linear combination of the expanatory variables:
        # t = w0 + x1 * w1 + x2 * w2 + ... + xn * wn
        t = (linear_combination(repository_record, self.model))
        return sigmoid(t) > 0.5

    def set_weights(self, weights):
        self.model = weights

    def serialize_to_file(self, path_to_model):
        with io.open(path_to_model, 'w', encoding="utf8") as json_file:
            json_file.write(json.dumps(self.model.tolist()))

    def deserialize_from_file(self, path_to_model):
        with io.open(path_to_model, 'r', encoding="utf8") as json_file:
            self.model = np.array(json.loads(json_file.readline()))

def connect_with_cache(db_path, cache_path):
    con = sqlite3.connect(db_path)
    con.execute("ATTACH \"{}\" AS CACHE".format(cache_path))
    return con

class LogisticModeling(Modeling):
    def __init__(self, preproc_dataset_path, cache_path):
        super().__init__(preproc_dataset_path)
        self.cache_path = cache_path

    def run_modeling(self, cross_validation_params):
        print("Starting parameter building")
        con = connect_with_cache(self.preproc_dataset_path, self.cache_path)

        con.execute("""
            CREATE TEMP VIEW selection AS
            SELECT repository.*
            FROM repository JOIN repo on repositoryID = id
            WHERE julianday(creation_date) <= julianday('{}') - 62
        """.format('2017-03-27'))
        data = np.array(gather_repo_data(con, TRAINING_TIME))
        activities = np.array(judge_active(con, LATEST_RECORD))

        # randomize both arrays while keeping the indices together
        assert(len(data) == len(activities))
        p = np.random.permutation(len(data))
        data = data[p]
        activities = activities[p]
        if args.sample:
            cutoff = math.ceil(len(data) * args.sample)
            data = data[:cutoff]
            activities = activities[:cutoff]

        training_cutoff = math.ceil(len(data) * TRAINING_FRACTION)
        x = data[:training_cutoff]
        y = activities[:training_cutoff]
        print("Done building parameters, starting modeling")
        model = logistic_regression(x, y)
        m = LogisticModel()
        m.set_weights(model)
        print("Done modeling, starting test")
        x = np.array(data[training_cutoff:])
        y = np.array(activities[training_cutoff:])
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for (xi, yi) in zip(x, y):
            predicted_event = m.regression_on_repository(xi)
            if yi == 1:
                if predicted_event:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if predicted_event:
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
        return m

# calculates the logistic function
def sigmoid(t):
    return 1 / (1 + sympy.exp(-t))

def linear_combination(x, a):
    t = a[0] # intercept
    for i in range(len(x)):
        t += x[i] * a[i + 1]
    return t

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

    return weight

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
def gather_repo_data(con, current_time, id = None):
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
            avg_release,
            avg_issue_created,
            avg_contrib,
            avg_contrib_1,
            avg_contrib_2
        FROM \"{}\"
        {}
    """.format(avg_name, "ORDER BY id" if id is None else "WHERE id = {}".format(id))).fetchall()
    deltas = con.execute("""
        SELECT
            delta_star,
            delta_push,
            delta_pr_created
        FROM \"{}\"
        {}
    """.format(delta_name, "ORDER BY id" if id is None else "WHERE id = {}".format(id))).fetchall()
    return np.append(averages, deltas, 1)

def train(args):
    modeling = LogisticModeling(args.dataset, args.cache)
    model = modeling.run_modeling(None)
    model.serialize_to_file(args.model)

def predict(args):
    model = LogisticModel()
    model.deserialize_from_file(args.model)
    result = model.regression_on_repository(json.loads(args.repo))
    if result:
        print("Prediction: This repository will be active in half a year.".format())
    else:
        print("Prediction: This repository will *not* be active in half a year.".format())

def predict_by_id(args):
    con = connect_with_cache(args.dataset, args.cache)
    model = LogisticModel()
    model.deserialize_from_file(args.model)
    data = gather_repo_data(con, TRAINING_TIME, id = args.id)[0]
    print("Predicting for repository: {}".format(data))
    print("Prediction: {}".format(model.regression_on_repository(data)))

if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a logistic regression model of GitHub repositories.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    parser.add_argument("-c", "--cache", help="Path to cache database.")
    parser.add_argument("-m", "--model", help="Model file (json) to read or write -- always required!")
    parser.add_argument("-p", "--predict", help="Predict the activity of a repository; takes in 11 numerical arguments and gives back 0 or 1")

    subparsers = parser.add_subparsers()

    # create the parser for the "train" command
    parser_a = subparsers.add_parser('train', help='Train the model on a dataset and serialize it to file')
    parser_a.add_argument("-s", "--sample", type=float, help="Use only sample% of the dataset")
    parser_a.set_defaults(func=train)

    # create the parser for the "predict" command
    parser_b = subparsers.add_parser('predict', help='Predict the activity of a repository (based on a deserialized model)')
    parser_b.add_argument('--repo', help='Record in format "[avg_star, avg_push, avg_pr_created, avg_release, avg_issue_created, contrib, contrib_1, contrib_2, delta_star, delta_push, delta_pr_created]"')
    parser_b.set_defaults(func=predict)

    # create the parser for the "predict_by_id" command
    parser_c = subparsers.add_parser('predict_by_id', help='Predict the activity of a repository in the dataset, given by its id')
    parser_c.add_argument('--id', type=int, help='Repository id')
    parser_c.set_defaults(func=predict_by_id)

    args = parser.parse_args()
    args.func(args)
