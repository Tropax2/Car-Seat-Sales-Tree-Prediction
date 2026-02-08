from sklearn.ensemble import RandomForestRegressor

def forest(n_estimators, 
           criterion, 
           max_depth, 
           max_features, 
           ccp_alpha,
           random_state = 42
           ):
    return RandomForestRegressor(n_estimators=n_estimators,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 ccp_alpha=ccp_alpha,
                                 random_state=random_state                               
                                 )
