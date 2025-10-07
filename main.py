from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from fpdf import FPDF
import io
import os
from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

import mysql.connector

def create_patients_table():
    db = mysql.connector.connect(
        host='your_mysql_host',
        user='your_user',
        password='your_password',
        database='your_database'
    )
    cursor = db.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100),
        username VARCHAR(50) UNIQUE,
        password VARCHAR(255),
        age INT,
        gender VARCHAR(10),
        ethnicity VARCHAR(20),
        smoking_status VARCHAR(10),
        alcohol_consumption_per_week FLOAT,
        physical_activity_minutes_per_week INT,
        diet_score INT,
        sleep_hours_per_day FLOAT,
        screen_time_hours_per_day FLOAT,
        family_history_diabetes TINYINT(1),
        hypertension_history TINYINT(1),
        cardiovascular_history TINYINT(1),
        bmi FLOAT,
        waist_to_hip_ratio FLOAT,
        systolic_bp INT,
        diastolic_bp INT,
        heart_rate INT,
        cholesterol_total FLOAT,
        hdl_cholesterol FLOAT,
        ldl_cholesterol FLOAT,
        triglycerides FLOAT,
        glucose_fasting FLOAT,
        glucose_postprandial FLOAT,
        insulin_level FLOAT,
        hba1c FLOAT,
        diabetes_risk_score INT,
        diabetes_stage VARCHAR(20),
        diagnosed_diabetes TINYINT(1)
    );
    """

    cursor.execute(create_table_query)
    db.commit()
    cursor.close()
    db.close()



REPO_ID = "realsarah87/randomforestdiabetes"
pipeline_path = hf_hub_download(repo_id=REPO_ID, filename="pipeline_model.pkl")


pipeline = joblib.load(pipeline_path)
y_encoder = joblib.load("label_encoder.pkl")

last_user_data = {}


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db = create_patients_table()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patients WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        db.close()

        if user and check_password_hash(user['password'], password):
            return f"Welcome {user['name']}! Login successful."
        else:
            return "Invalid username or password!"

        return redirect(url_for('success'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        db = create_patients_table()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO patients (name, username, password) VALUES (%s, %s, %s)",
                           (name, username, password))
            db.commit()
            msg = "Signup successful!"
        except mysql.connector.IntegrityError:
            msg = "Username already exists!"
        cursor.close()
        db.close()
        return msg

        return redirect(url_for('success'))

    return render_template('signup.html')



@app.route('/success', methods=['GET', 'POST'])
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
