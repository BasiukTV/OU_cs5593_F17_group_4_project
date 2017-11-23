from flask_wtf import Form 
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import DataRequired

class PreprocessingForm(Form):
    fpath = StringField('Dataset Path:', validators=[DataRequired("Please enter the dataset path.")])
    submit = SubmitField('Proceed')

class ClusteringForm(Form):
    clusteringAlgorithms = RadioField('Clustering Algorithms:', choices=[(1, 'Simple K-means'), (2, 'Hierarchical')])
    submit = SubmitField('Proceed')

class RegressionForm(Form):
    regressionAlgorithms = RadioField('Regression Algorithms:', choices=[(1, 'Logistic Regression'), (2, 'Linear Regression')])
    submit = SubmitField('Proceed')
