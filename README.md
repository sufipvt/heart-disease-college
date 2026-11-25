# Experiment 7: Deployment

## Objective
Deploy the selected heart-disease classification pipeline as a simple Streamlit decision-support demonstration.

## Files
- `app.py`: Streamlit application.
- `README.md`: setup and usage instructions.
- `sample_prediction_log.csv`: created when the first prediction is made.

## Setup
1. Run Experiment 5 first so the saved model artifacts exist.
2. Use the results from Experiment 6 to choose the model to deploy.
3. In `app.py`, set `MODEL_FILE` to the selected artifact:
   - `logistic_regression.pkl`
   - `random_forest.pkl`
   - `svm.pkl`
4. Put that `.pkl` file in the same folder as `app.py`. It must be the complete pipeline saved in Experiment 5, including preprocessing.

Install packages in the terminal:

```bash
pip install streamlit pandas scikit-learn joblib
```

Run the app:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

## How to use
1. Enter values using the dataset's feature definitions and units.
2. Click **Predict**.
3. Read the predicted class and optional class-1 model score.
4. The app appends a sample input/output row to `sample_prediction_log.csv`.

## Sample request
`age=55, sex=1, cp=0, trestbps=130, chol=240, fbs=0, restecg=1, thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=2`

## Sample response format
The app returns the predicted class (0 or 1) and, when supported, the model's class-1 score. Exact results depend on the trained model; run the app to obtain the real output. Do not invent a prediction for the report.

## Limitations and safety
- This is an educational prototype, not a medical device.
- Do not use it to diagnose, rule out, or treat heart disease.
- The displayed score is not a clinically validated individual risk estimate.
- The dataset is small and may not represent other populations or current clinical practice.
- Do not enter patient names, contact details, IDs, or other identifying information.
- A qualified healthcare professional must interpret any output.
