import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Price'] = data.target
    return df
