A machine-learning project exploring whether a flight is classified as
delayed or not delayed using historical flight information.
Project overview
This project uses a historical flight dataset to explore patterns
associated with flight delays and compare two binary-classification
models:
- Logistic Regression
- Random Forest Classifier
The target column is is_delay, where 0 represents the not-delayed
class and 1 represents the delayed class in the supplied notebook.
Dataset
The notebook loads flight_delay_predict.csv with Pandas.

  Property                        Value
  Rows                        1,635,590
Original columns                   20
Training split                    80%
Test split                        20%
Delayed class (1)           343,332
Not-delayed class (0)     1,292,258

The columns include calendar information, reporting airline, origin and
destination airports/states, scheduled departure time, distance,
airtime, and delay-related fields.

The notebook's observed data contains five airline codes and nine
origin/destination airport codes. The source and exact sampling criteria
for the CSV should be documented if known.
Workflow
1. Load and inspect data with Pandas (head, shape, info,
   describe, value_counts, and nunique).
2. Explore delay patterns, including delay rates by month and
   origin airport.
3. Prepare features and target:
   - Set is_delay as the target.
   - Remove Cancelled, Diverted, ArrDelay, and
     ArrDelayMinutes from the model input.
   - Remove the string-formatted FlightDate column.
   - Convert categorical columns (airline, origin/destination airport
     and state) into indicator columns using
     pandas.get_dummies(drop_first=True).
4. Split the data into training and test sets using an 80/20 random
   split with random_state=42.
5. Train and evaluate Logistic Regression and Random Forest
   classifiers.
6. Inspect model performance using accuracy, per-class
   precision/recall/F1, confusion matrices, and Random Forest feature
   importances.
7. Experiment with class weighting by training a Random Forest with
   class_weight="balanced".
Model results
Results below are copied from the notebook's recorded test-set outputs.

  Model                                              Accuracy   Delayed-flight recall (class 1)
  Logistic Regression                                  79.57%                                5%
Random Forest                                        81.11%                               25%
Random Forest (class_weight="balanced")     81% (rounded)                               23%

Random Forest achieved the highest recorded accuracy. However, its
delayed-flight recall was only 25%, meaning it detected approximately
one quarter of the delayed flights in the test set. The dataset is
imbalanced: approximately 21% of rows belong to class 1.

Interpretation: accuracy alone is not sufficient for this task.
Recall, precision, F1-score, and the confusion matrix are important when
evaluating the model's ability to identify delayed flights.
Important limitations
- Potential prediction-time leakage: the feature AirTime is
  actual flight duration and is not normally available before
  departure. A model using it cannot be presented as a true
  pre-departure prediction system.
- Random split: the notebook uses a random split, not a
  chronological evaluation. A stronger forecasting evaluation would
  train on earlier dates and test on later dates.
- Target definition: the notebook uses an existing is_delay
  column, but the supplied notebook extract does not document how that
  label was originally calculated. The delay threshold and
  label-generation method should be added when confirmed.
- Class imbalance: delayed flights are the minority class, and the
  models have relatively low recall for that class.
- Preprocessing reproducibility: categorical encoding is performed
  before the train/test split. A production-quality workflow should
  fit preprocessing only on training data, ideally using a
  scikit-learn Pipeline and ColumnTransformer.
- Logistic Regression convergence: the recorded run produced a
  convergence warning at 1,000 iterations. Scaling numerical features
  and tuning the model should be investigated.
- The notebook is an exploratory analysis, not a deployed or
  operational airline prediction service.
Technologies
- Python
- Pandas
- NumPy
- scikit-learn
- Jupyter Notebook
Running the notebook
1. Clone this repository.
2. Install the required packages:
   pip install pandas numpy scikit-learn jupyter
3. Place flight_delay_predict.csv in the location expected by the
   notebook, or update the CSV path.
4. Launch Jupyter:
   jupyter notebook
5. Open and run the Flight Delay Prediction notebook.
The dataset itself may not be included in the repository. Check its
source and licensing before redistributing it.

Possible next improvements
- Verify and document the exact definition of is_delay.
- Remove features unavailable at the intended prediction time,
  especially actual AirTime.
- Use a chronological train/validation/test split.
- Move encoding and scaling into a leakage-safe scikit-learn pipeline.
- Tune the classification threshold and evaluate precision-recall
  trade-offs.
- Compare class-weighting and resampling approaches using
  delayed-class recall and F1-score.
- Add a reproducible requirements file and save evaluation plots.
