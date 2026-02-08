from sklearn.ensemble import BaggingRegressor 
from sklearn.tree import DecisionTreeRegressor as DTR 

# define the bagging method with the optimal values obtained from a single tree approach
def bagging(n_estimators,
            max_depth,
            max_features,
            ccp_alpha,
            random_state = 42
             ):
    return BaggingRegressor(estimator=DTR(max_depth=max_depth, ccp_alpha=ccp_alpha), 
                            n_estimators=n_estimators, 
                            max_features=max_features, 
                            random_state=random_state)

