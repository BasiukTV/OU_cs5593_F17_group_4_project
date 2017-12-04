from flask_wtf import Form
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField, RadioField, TextField
from wtforms.validators import DataRequired

class PreprocessingForm(Form):
    fpath = FileField('Your JSON File: ', validators=[DataRequired("Please enter the dataset path.")])
    submit = SubmitField('Proceed')

class ClusteringForm(Form):
    clusteringAlgorithms = RadioField('Available Clustering Algorithms:', choices=[(1, 'Simple K-means'), (2, 'Hierarchical')], default = 1)
    cliArguments = TextField("CLI Arguments ")
    submit = SubmitField('Proceed')

class RegressionForm(Form):
    regressionAlgorithms = RadioField('Regression Algorithms:', choices=[(1, 'Logistic Regression'), (2, 'Linear Regression')], default = 1)
    submit = SubmitField('Proceed')
    repositoryID = TextField("Repository ID ")
    avg_star = TextField("Average star ")
    avg_push = TextField("Average push ")
    avg_pr_created = TextField("Average PR created ")
    avg_release = TextField("Average release ")
    avg_issue_created = TextField("Average issue created ")
    avg_contrib = TextField("Contributors ")
    avg_contrib_1 = TextField("Contributors T1 ")
    avg_contrib_2 = TextField("Contributors T2 ")
    delta_star = TextField("Delta star ")
    delta_push = TextField("Delta push ")
    delta_pr_created = TextField("Delta PR created ")
