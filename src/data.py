import pandas as pd 
from sklearn.model_selection import train_test_split
# Transform the dataset into a pandas df and remove rows with null values 
def csv_to_df(csv_path : str) -> pd.DataFrame:
    Sales = pd.read_csv(csv_path)
    Sales = Sales.dropna()

    return Sales 

# Split the data into a training and test set
def make_splits(
        X,
        y,
        test_size = 0.2,
        random_state = 42
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)