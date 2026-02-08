import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer

# one-hot-encode categorical predictors and scale numerical ones
def transformer(
        categorical_predictors: list[str],
        numerical_predictors: list [str]
):
    numeric = StandardScaler()
    enc = OneHotEncoder(drop="first", handle_unknown="ignore")

    prepocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numerical_predictors),
            ("cat", enc, categorical_predictors)
        ],
        remainder="drop"
    )
    return prepocessor

    