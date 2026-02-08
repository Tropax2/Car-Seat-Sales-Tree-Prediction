from sklearn.ensemble import GradientBoostingRegressor as GBR 

def boosting(loss, 
             n_estimators, 
             learning_rate, 
             max_features,
             max_depth, 
             ccp_alpha, 
             random_state=42):
    
    return GBR(loss=loss, 
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                learning_rate=learning_rate,
                ccp_alpha=ccp_alpha,
                random_state=random_state
               )