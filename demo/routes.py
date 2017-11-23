from flask import Flask, render_template, request, redirect, url_for
from forms import DatasetForm

app = Flask(__name__)

app.secret_key = "development-key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    form = DatasetForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('fpath.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('fpath.html', form=form)

@app.route("/kmeans", methods=["GET"])
def kmeans():
    # place holder
    form = DatasetForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('fpath.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('fpath.html', form=form)

@app.route("/hierarchical", methods=["GET"])
def hierarchical():
    # place holder
    form = DatasetForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('fpath.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('fpath.html', form=form)

@app.route("/linearReg", methods=["GET"])
def linearReg():
    # place holder
    form = DatasetForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('fpath.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('fpath.html', form=form)

@app.route("/logisticReg", methods=["GET"])
def logisticReg():
    # place holder
    form = DatasetForm()

    if request.method == "GET":
        if form.validate() == False:
            return render_template('fpath.html', form=form)
        else:
            return redirect(url_for('home'))

    elif request.method == "GET":
        return render_template('fpath.html', form=form)

if __name__ == "__main__":
  app.run(debug=True)
