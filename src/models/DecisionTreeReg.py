from sklearn.tree import DecisionTreeRegressor as DTR 

def decision_tree(max_depth, ccp_alpha = 0):
    return DTR(criterion="squared_error", max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=42)
