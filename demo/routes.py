from flask import Flask, render_template, request, redirect, url_for
from forms import PreprocessingForm, ClusteringForm, RegressionForm

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

    if request.args.get('cliArguments', '') and request.args.get('clusteringAlgorithms','') == "2":
        return "I might do hierarchical clustering for you in future ..."

    if request.method == "GET":
        if form.validate() == False:
            return render_template('cluster.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('cluster.html', form=form)

@app.route("/regress", methods=["GET"])
def regress():
    form = RegressionForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('regress.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('regress.html', form=form)

if __name__ == "__main__":
  app.run(debug=True)
