# Motion-based Activity Recognization
A machine learning pipeline to recognize different activities using a phone's motion sensors (accelerometer).

## Data collection
The activities we are looking for include walking, climbing up the stairs, limbing down the stairs, and squats. The data is collected with *Sensor Logger* with frequency 50Hz. We have gathered 10 users for data collection. The approximate amount of training data collected for each user is listed below.  
- walking: 2 hours
- climbing up the stairs: 15 minutes
- climbing down the stairs: 15 minutes
- squats: 50 instances

Here's a link to [data](https://drive.google.com/drive/folders/1ngiT8UwPam8lQSS_eyJfPGKvpgp0pr86) we have collected.

## Extract Features
### Step 1: Dividing the data into 10 seconds intervals
Since the data collected is a continuous stream, so I've divided data into 10 seconds interavals to extract features, i.e. we're making decisions every 10 seconds. The data is collected at 50Hz, which means each interval should contain 500 rows of input.

### Step 2: Extract features
Observe the visualizations we have before, the main difference is that walking downstairs and upstairs should have a more significant accerlation on the vertical axis, while walking should be relatively stable. Using x, y, z values from accelerometer, the following features are extracted within each sample:
- Mean, std, max, min of x
- Mean, std, max, min of y
- Mean, std, max, min of z

## Model training
As mentioned before, considering the data imbalance between different classes, I used `RandomeForestClassifier` with `class_weight = ‘balanced’` for training.

## Evaluation
The pipeline's performance is analyzed using 10-fold cross-validation.The model seems to perfrom fairly well.

We can observe that walking seems to be more accurate than other events. That might happen mainly because we have way more walking data compared to other events. 
