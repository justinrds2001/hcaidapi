from flask import Flask, jsonify, render_template, request
from whitenoise import WhiteNoise  # Import WhiteNoise
import pandas as pd
import shap
import base64
import io
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import bz2

def fix_json(obj):
    return {
        "age": obj["age"],
        "hypertension": obj["hypertension"],
        "heart_disease": obj["heart_disease"],
        "bmi": obj["bmi"],
        "HbA1c_level": obj["HbA1c_level"],
        "blood_glucose_level": obj["blood_glucose_level"],
        "gender_encoded": obj["gender_encoded"],
        "smoking_history_encoded": obj["smoking_history_encoded"]
    }

try:
    good_model = tf.keras.models.load_model("static/diabetes_good_model.h5")
    bad_model = tf.keras.models.load_model('static/diabetes_bad_model.h5')
except Exception as e:
    print("model loading error:", e)
    exit()

try:
    # with open("static/explainer_bad.pkl", "rb") as explainer1:
    #     bad_explainer = pickle.load(explainer1)
    with open("static/explainer_good.pkl", "rb") as explainer2:
        good_explainer = pickle.load(explainer2)
    # bad_explainer = joblib.load("static/explainer_bad.bz2")
    # good_explainer = joblib.load("static/explainer_good.bz2")
except Exception as e:
    print("explainer loading error:", e)
    exit()

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictgood", methods=['POST'])
def do_prediction_good():
    json_data = request.get_json()
    json_data = fix_json(json_data)
    df = pd.DataFrame(json_data, index=[0])

    # predict
    shap_values = good_explainer.shap_values(df)
        
    y_pred = good_model.predict(df)
    pred_diabetes = int(y_pred[0])
    
    # shap
    i = 0
    shap.force_plot(good_explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    # Return the Base64-encoded image string in the response
    result_map = {0: False, 1: True}
    print(result_map[pred_diabetes])
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

@app.route("/predictbad", methods=['POST'])
def do_prediction_bad():
    json_data = request.get_json()
    json_data = fix_json(json_data)
    df = pd.DataFrame(json_data, index=[0])

    # predict

    y_pred = bad_model.predict(df)
    pred_diabetes = int(y_pred[0])
    shap_values = bad_explainer.shap_values(df)
    i = 0
    shap.force_plot(bad_explainer.expected_value[i], shap_values[i], df.iloc[i], matplotlib=True, show=False, plot_cmap=['#77dd77', '#f99191'])

    # Save the plot as a Base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode("utf-8")

    result_map = {0: False, 1: True}
    return jsonify({'diabetes': result_map[pred_diabetes], 'image_base64': base64_image})

if __name__ == '__main__':
  app.run(port=5000)


