# mtb-IA

Tool to make predictions about MTB rides

# Preprocessing

1. Get a Strava dump of your rides, save it in data/strava_export
2. Run strava_dump_decompressor.py
3. Run strava_activity_processor.py
4. Run preprocess.py. The result is processed_activities.csv. This file contains all the waypoints of all the rides in
   the Strava dump.

# Working models

## Route completion time prediction

Predicts the time it will take to complete a route based on the distance and elevation information.

- preprocess.py: Reads the processed_activities.csv file and creates a new file with the relevant information for the
  model.
- train_model.py: Trains various model using the data from the previous step. Currently, the models are:
    - Ridge regression
    - SVR regression
    - MLP regression
- predict.py: Reads GPX route files and predicts the time it will take to complete them using the best models.

## Training the Regressor

The training of the regressor is handled in the `train_regressor.py` script. This script reads the preprocessed data, prepares it for training, and then trains various regression models on it. The models currently supported are Ridge, SVR Linear, SVR RBF, Lasso, and ElasticNet.

The script uses a pipeline to preprocess the data and train the model. The preprocessing step includes standard scaling for numerical features and one-hot encoding for categorical features. The training step uses GridSearchCV for hyperparameter tuning.

The script splits the data into a training set and a test set. The model is trained on the training set, and its performance is evaluated on the test set. The performance metrics used are Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).

After training, the script saves the model, its performance metrics, and its coefficients to the `model_stats` directory. It also plots the coefficients of the model.

To train a regressor, run the `train_regressor.py` script. You can adjust the parameters of the models and the features used for training at the top of the script.