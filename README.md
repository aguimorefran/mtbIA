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