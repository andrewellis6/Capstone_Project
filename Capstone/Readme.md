
# Massachusetts Education Capstone Project By: Andrew Ellis

Despite being the richest country in the world, America ranks barely above average in reading and science, and below average in math according to the PEW Research center. My goal is to use public data from Massachusetts to idenitfy what features lead to high student outcomes at ALL levels. I chose this dataset from Kaggle because it was pre-merged and had over 300 features that I could analyze. However, after doing some initial EDA, I realized that there was still a significant amount of data-cleaning that needed to happen if I was going to run models that would give me proper insight. I am wondering what superintendents can do to turn around a school district.

# Part 1 - EDA and Visualization

To start, these are the libraries and software supports (I am using a mac...)


```python
!pip install -U pandasql;
!pip install --upgrade pip

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN
from pandasql import sqldf

from matplotlib import cm
from collections import OrderedDict

cmaps = OrderedDict()

%matplotlib inline
```

    Requirement already up-to-date: pandasql in /opt/conda/lib/python3.6/site-packages
    Requirement already up-to-date: numpy in /opt/conda/lib/python3.6/site-packages (from pandasql)
    Requirement already up-to-date: sqlalchemy in /opt/conda/lib/python3.6/site-packages (from pandasql)
    Requirement already up-to-date: pandas in /opt/conda/lib/python3.6/site-packages (from pandasql)
    Requirement already up-to-date: pytz>=2011k in /opt/conda/lib/python3.6/site-packages (from pandas->pandasql)
    Requirement already up-to-date: python-dateutil>=2 in /opt/conda/lib/python3.6/site-packages (from pandas->pandasql)
    Requirement already up-to-date: six>=1.5 in /opt/conda/lib/python3.6/site-packages (from python-dateutil>=2->pandas->pandasql)
    Requirement already up-to-date: pip in /opt/conda/lib/python3.6/site-packages


After importing the necessary libraries, I converted the kaggle data set from: https://www.kaggle.com/ndalziel/massachusetts-public-schools-data/data to a pandas dataframe to make it easier to manipulate. I was able to identify columns with high counts of null values and features that will not lead to any predictive power (like fax number).

After initially having trouble deciding on a target variable, I decided to split the data up and create dataframes specific to Elementary, Middle, and High Schools respectively. To do this, I had to think of what each type of school would contain that is unique to that school. I decided to split the data on certain grade levels having columns. I chose 2nd for Elementary, 7th for Middle School, and 10th for High School.

It will take a long time to get my 3 datasets to a place where they can be analyzed... but using the thresh parameter on the dropna method has sped up the process.
