import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

wellness_path = "../data/train/wellness.csv"

data = pd.read_csv(wellness_path)

# Date col is format 2023-02-14
data["date"] = pd.to_datetime(data["date"])

# Add a new column for the weight.
#

print(data.head())