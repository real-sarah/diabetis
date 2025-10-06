from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from fpdf import FPDF
import io
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)


REPO_ID = "realsarah87/randomforestdiabetes"
pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="pipeline_model.pkl")


pipeline = joblib.load(pipeline_path)
y_encoder = joblib.load("label_encoder.pkl")

last_user_data = {}

# ----------------------------
# Routes
# ----------------------------

@app.route('/', methods=['GET', 'POST'])
def home():
    global last_user_data
    if request.method == 'POST':
        data = request.form.to_dict()
        last_user_data = data.copy()

        # Convert numeric values
        for key in data:
            try:
                data[key] = float(data[key])
            except ValueError:
                pass

        df = pd.DataFrame([data])


        for col in pipeline.feature_names_in_:
            if col not in df.columns:
                df[col] = 0

        df = df[pipeline.feature_names_in_]

        # Prediction
        pred_encoded = pipeline.predict(df)[0]
        pred_label = y_encoder.inverse_transform([pred_encoded])[0]

        return render_template('result.html', pred_label = pred_label)
    return render_template('index.html')


@app.route('/download/<prediction>')
def download(prediction):
    global last_user_data
    data = last_user_data or {}

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt='Diabetes Diagnosis Report', ln=True, align='C')
    pdf.ln(10)

    pdf.set_font('Arial', '', 14)
    pdf.cell(200, 10, txt=f'Prediction: {prediction}', ln=True, align='L')
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 8, txt='Entered Details:', ln=True)
    pdf.ln(5)

    pdf.set_font('Arial', '', 12)
    for key, value in data.items():
        nice_key = key.replace('_', ' ').title()
        pdf.cell(200, 8, txt=f"{nice_key}: {value}", ln=True)

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='diagnosis_report.pdf'
    )


if __name__ == '__main__':
    app.run(debug=True)
