from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from fpdf import FPDF
import io
import os
from huggingface_hub import hf_hub_download



app = Flask(__name__)


REPO_ID = "realsarah87/randomforestdiabetes"
pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="pipeline_model2.pkl")


pipeline = joblib.load(pipeline_path)
y_encoder = joblib.load("label_encoder.pkl")

last_user_data = {}


@app.route('/', methods=['GET', 'POST'])
def home():
    global last_user_data
    if request.method == 'POST':
        data = request.form.to_dict()
        last_user_data = data.copy()


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


        pred_encoded = pipeline.predict(df)[0]
        pred_label = y_encoder.inverse_transform([pred_encoded])[0]

        return render_template('result.html', pred_label = pred_label)
    return render_template('index.html')


@app.route('/download/<prediction>')
def download(prediction):
    global last_user_data
    data = last_user_data

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_fill_color(0, 51, 102)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, txt='  DIABETES DIAGNOSIS REPORT', ln=True, align='L', fill=True)
    pdf.ln(2)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 16)
    # pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
    pdf.ln(5)

    is_high_risk = 'Type 2' in prediction.lower() or 'Type 1' in prediction.lower() or 'Gestational' in prediction.lower() or 'error' in prediction.lower()
    if is_high_risk:
        fill_rgb = (255, 236, 236)
        border_rgb = (231, 76, 60)
        text_rgb = (192, 57, 43)
    else:
        fill_rgb = (220, 255, 220)
        border_rgb = (46, 204, 113)
        text_rgb = (39, 174, 96)

    pdf.set_fill_color(*fill_rgb)
    pdf.set_draw_color(*border_rgb)
    pdf.set_text_color(*text_rgb)
    pdf.set_font('Helvetica', 'B', 30)
    pdf.cell(0, 18, txt=f'  {prediction.upper()}', border=1, ln=True, align='C', fill=True)
    pdf.ln(10)

    # --- Patient Data ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_fill_color(204, 204, 204)
    pdf.cell(0, 10, txt='  Patient Input Details', ln=True, fill=True)
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 12)
    for key, value in data.items():
        nice_key = key.replace('_', ' ').title()
        pdf.cell(90, 7, txt=f"{nice_key}:", border=0)
        pdf.cell(0, 7, txt=str(value), ln=True)

    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 5,
                   txt="* Disclaimer: This is an automated report. Please consult a healthcare provider for medical advice. *",
                   border=1, align='C')

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name='diagnosis_report.pdf')

if __name__ == '__main__':
    app.run(debug=True)