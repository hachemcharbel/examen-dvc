import pandas as pd
import pickle
import os
import json
from sklearn.metrics import mean_squared_error, r2_score

with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').squeeze() 

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

scores = { 'mean_squared_error': mse, 'r2_score': r2}

predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred
})
predictions_df.to_csv('data/predictions.csv', index=False)

with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print("Évaluation terminée ")
