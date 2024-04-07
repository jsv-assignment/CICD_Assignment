import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import numpy as np

import xgboost as xgb

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Running XGBoost instead of Logistic Regression to improve the accuracy
#model = LogisticRegression().fit(X, y)
model = xgb.XGBClassifier().fit(X, y)


with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)

