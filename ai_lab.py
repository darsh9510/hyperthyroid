!pip install pgmpy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv('allhyper.data', delimiter=',', header=None)
data.columns = ['hyperthyroid', 'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source']

X = data.drop('hyperthyroid', axis=1)
y = data['hyperthyroid']

X = pd.get_dummies(X)  
y = y.astype('category').cat.codes  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)

model = BayesianNetwork([
    ('age', 'hyperthyroid'), 
    ('sex', 'hyperthyroid'), 
    ('on thyroxine', 'hyperthyroid')
])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

predictions = []
for index, row in X_test.iterrows():
    evidence = row.to_dict()
    evidence = {key: evidence[key] for key in evidence if key in model.nodes()}
    query = inference.map_query(variables=['hyperthyroid'], evidence=evidence)
    predictions.append(query['hyperthyroid'])

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the model is: {accuracy * 100:.2f}%")