from flask_wtf import Form 
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class DatasetForm(Form):
    fpath = StringField('File Path', validators=[DataRequired("Please enter the dataset path.")])
    submit = SubmitField('Train Model')

