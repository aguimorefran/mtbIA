import pandas as pd

data = pd.read_csv("preprocessed.csv")

print("Shape of the data: ", data.shape)

# Preprocessing
# Save col names with their index
# Remove idx=0 and idx=1 cols
data = data.drop(data.columns[[0, 1]], axis=1)
print("Shape of the data after removing idx=0 and idx=1: ", data.shape)
print("Columns of the data: ", data.columns)


def train_ridge_model(df):


    pass

def train_svr_model(df):


    pass

def train_mlp_model(df):


    pass