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
