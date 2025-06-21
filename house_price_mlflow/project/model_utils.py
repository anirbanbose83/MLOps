from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning(X_train, y_train, param_grid):
    model = RandomForestRegressor()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid
