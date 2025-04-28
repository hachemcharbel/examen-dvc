import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').squeeze()  

model = RandomForestRegressor(random_state=42)

# Définir la grille des hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

scorer = make_scorer(mean_squared_error, greater_is_better=False)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,        
    n_jobs=-1,    
    verbose=2
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)

with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

