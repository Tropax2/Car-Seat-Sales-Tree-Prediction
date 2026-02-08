import numpy as np
import pandas as pd
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error
import data 
import features 
from models import DecisionTreeReg, bagging, random_forest, boosting
import cv 

# This is the main function
def main():
    # Get the file location 
    path = r'path'

    # Transform into a df 
    Sales = data.csv_to_df(path)

    # Separate the predictors and the response
    numerical = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]
    categorical = ["ShelveLoc", "Urban", "US"]
    predictors = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education", "ShelveLoc", "Urban", "US"]
    X = Sales[predictors]
    response = "Sales"

    # Separate the dataset 
    X_train, X_test, y_train, y_test = data.make_splits(X = X, y = Sales[response], random_state=42)

    # Transform the predictors 
    preprocessor = features.transformer(categorical_predictors=categorical, numerical_predictors=numerical)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Fit the regression tree to the training data 
    clf = DecisionTreeReg.decision_tree(max_depth=5, ccp_alpha=0.0567)
    clf.fit(X_train_transformed, y_train)

    # Compute the test MSE 
    mse = mean_squared_error(clf.predict(X_test_transformed), y_test)
    print(mse)

    # Cross-validate
    ccp_path = clf.cost_complexity_pruning_path(X_train_transformed, y_train)
    grid_search = cv.grid(estimator = clf, 
                          param_grid = {
                            'max_depth': [i for i in range(1, 30)],
                            "ccp_alpha": ccp_path.ccp_alphas   
                                     })
    grid_search.fit(X_train_transformed, y_train)
    best_ = grid_search.best_estimator_
    print(best_)    

    # Fit the bagging to the training data  
    regr = bagging.bagging(n_estimators=500, 
                           max_depth=6,
                           max_features=X_train_transformed.shape[1],
                           ccp_alpha=0.05
                           )
    regr.fit(X_train_transformed, y_train)

    # Compute the test MSE
    mse_bag = mean_squared_error(regr.predict(X_test_transformed), y_test)
    print(mse_bag)

    # Fit the random forest to the training data 
    regrf = random_forest.forest(n_estimators=500, 
                                 criterion="squared_error", 
                                 max_depth=6, 
                                 max_features=4,
                                 ccp_alpha=0.05
                                 )
    regrf.fit(X_train_transformed, y_train)
    
    # Compute the test MSE 
    mse_for = mean_squared_error(regrf.predict(X_test_transformed), y_test)
    print(mse_for)

    # Obtain the most important predictors for random forest 
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [n.replace("num__", "").replace("cat__","") for n in feature_names]
    feature_imp = pd.DataFrame({"importance": regrf.feature_importances_}, index=feature_names)
    print(feature_imp.sort_values(by="importance", ascending=False))

    # Fit the boosting to the training data
    boost = boosting.boosting(
        loss="squared_error",
        n_estimators=500,
        max_depth=5,
        max_features=X_train_transformed.shape[1],
        learning_rate=0.05,
        ccp_alpha=0,
    )
    boost.fit(X_train_transformed, y_train)

    # Compute the test MSE 
    mse_boost = mean_squared_error(boost.predict(X_test_transformed), y_test)
    print(mse_boost)

if __name__ == "__main__":
    main()


