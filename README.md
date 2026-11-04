# Leaf Disease Classifier

A Streamlit app that classifies a single uploaded leaf photo with a trained 38-class TensorFlow model.

## Run locally

Use Python 3.11, then install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a clear photo of one leaf. The app shows the top prediction, its model score, and two alternatives. Scores are model outputs, not calibrated probabilities or a confirmed plant diagnosis.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, choose **Create app** and select this repository, branch `main`, and `app.py`.
3. In **Advanced settings**, select Python **3.11**. TensorFlow 2.15 does not support the default Python 3.12 runtime.

The trained model and class labels are included in this repository. The training dataset and exploratory notebook are not included.
