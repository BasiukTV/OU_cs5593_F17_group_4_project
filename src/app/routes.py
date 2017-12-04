from flask import Flask, render_template, request, redirect, url_for, flash
from forms import PreprocessingForm, ClusteringForm, RegressionForm
import os
import sys

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

app = Flask(__name__)
app.secret_key = "development-key"

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/preprocess", methods=["GET", "POST"])
def preprocess():
    form = PreprocessingForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('preprocess.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('preprocess.html', form=form)

@app.route("/cluster", methods=["GET"])
def cluster():
    form = ClusteringForm()

    # Connect to running Hierarchical clustering
    if request.args.get('cliArguments', '') and request.args.get('clusteringAlgorithms','') == "2":
        import argparse, os, sys
        from collections import namedtuple

        from modeling.clustering.hierarchical import HierarchicalModeling

        cli_args = request.args.get('cliArguments', '').split(' ')

        # Configuring CLI arguments parser and parsing the arguments
        parser = argparse.ArgumentParser("Script for creating a hierarchical model of GitHub contributors.")
        parser.add_argument("-d", "--dataset", required=True, help="Path to preprocessed dataset.")
        parser.add_argument("-ts", "--trial-size", type=int, required=True, help="Number of contributors to include in a trial. Hierarchical clustering uses O(n**2) of memory. This number cannot be very large.")
        parser.add_argument("-cdbp", "--clustering-db-path", help="Path to the output clustering db.")
        parser.add_argument("-ws", "--weights", nargs='+', type=float, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], help="Array of weights to be aplied to contributor attributes for purposes of determining distances between them.")
        arg = parser.parse_args(cli_args)

        # We use simple named tuple so we don't have to define a class which will not be used anywhere else
        RuntimeParameters = namedtuple('RuntimeParameters', ['trial_size'])

        modeling = HierarchicalModeling(arg.dataset)
        return modeling.run_modeling(RuntimeParameters(arg.trial_size), arg.weights, arg.clustering_db_path).__html__()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('cluster.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('cluster.html', form=form)

MODEL_PATH = '../../output/model/logistic.json'

@app.route("/regress", methods=["GET"])
def regress():
    form = RegressionForm()

    # print(request.args.get('regressionAlgorithms',''))
    # print(request.args.get('repositoryID',''))
    # Connect to running Hierarchical clustering
    if len(request.args.get('repositoryID', '') or '') > 0 and request.args.get('regressionAlgorithms','') == "1":
        from modeling.regression.logistic import by_id
        id = int(request.args.get('repositoryID'))
        (data, prediction) = by_id('../../input/data/repo-all.sqlite3', '../../input/data/repo-cache-all.sqlite3', MODEL_PATH, id)
        flash("Prediction for id {} ({}):".format(id, data))
        flash(str(prediction))
        return render_template('regress.html', form=form)
    elif request.args.get('avg_star', '') and request.args.get('regressionAlgorithms','') == "1":
        from modeling.regression.logistic import LogisticModel
        import json
        data = [
            float(request.args.get('avg_star')),
            float(request.args.get('avg_push')),
            float(request.args.get('avg_pr_created')),
            float(request.args.get('avg_release')),
            float(request.args.get('avg_issue_created')),
            float(request.args.get('avg_contrib')),
            float(request.args.get('avg_contrib_1')),
            float(request.args.get('avg_contrib_2')),
            float(request.args.get('delta_star')),
            float(request.args.get('delta_push')),
            float(request.args.get('delta_pr_created'))
        ]
        model = LogisticModel()
        model.deserialize_from_file(MODEL_PATH)
        prediction = model.regression_on_repository(data)
        flash("Prediction for {}:".format(data))
        flash(str(prediction))
        return render_template('regress.html', form=form)

    if request.method == "GET":
        if form.validate() == False:
            return render_template('regress.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('regress.html', form=form)

if __name__ == "__main__":
  app.run(debug=True)
