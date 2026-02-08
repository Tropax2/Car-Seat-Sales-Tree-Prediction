from sklearn.model_selection import GridSearchCV, KFold

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

# A 5-fold cross-validator that search values on a grid
def grid(estimator, param_grid):
    return GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)





