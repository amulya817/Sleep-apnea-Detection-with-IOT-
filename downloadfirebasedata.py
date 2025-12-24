# install Pyrebase4 first:
# pip install pyrebase4 pandas

import pyrebase
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Firebase Configuration
# ---------------------------------------------------
firebaseConfig = {
    "apiKey": "AIzaSyB9DlcQUBcLN9HLpAMg0NPe4MXinSqk-U8",
    "authDomain": "sleep-apnea-98274.firebaseapp.com",
    "databaseURL": "https://sleep-apnea-98274-default-rtdb.firebaseio.com",
    "projectId": "sleep-apnea-98274",
    "storageBucket": "sleep-apnea-98274.firebasestorage.app",
    "messagingSenderId": "509528365321",
    "appId": "1:509528365321:web:9d485df934a171ec661bd8",
    "measurementId": "G-DSP5152QEP"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# ---------------------------------------------------
# Target Data Path
# ---------------------------------------------------
patient_id = 10
data_path = f"SensorData/patient_{patient_id}"

# ---------------------------------------------------
# Apnea Detection Logic (1 = Possible Apnea, 0 = Normal)
# ---------------------------------------------------
def detect_apnea(df):
    df = df.copy()

    df["Spo2_prev"] = df["spo2"].shift(1)
    df["HR_prev"] = df["heart_rate"].shift(1)

    df["Spo2_drop"] = df["Spo2_prev"] - df["spo2"]
    df["HR_rise"] = df["heart_rate"] - df["HR_prev"]

    df["Apnea_Label"] = np.where(
        (df["Spo2_drop"] >= 4) & (df["HR_rise"] >= 10),
        1,
        0
    )

    return df.drop(columns=["Spo2_prev", "HR_prev", "Spo2_drop", "HR_rise"])

# ---------------------------------------------------
# Callback executed on new data arrival
# ---------------------------------------------------
def stream_handler(message):
    print("\n📌 New data received from Firebase!")
    print("Event:", message["event"])
    print("Path:", message["path"])
    print("Updated Data:", message["data"])

    snapshot = db.child(data_path).get()
    if not snapshot.each():
        print("⚠ No data found!")
        return

    readings = [item.val() for item in snapshot.each()]
    df = pd.DataFrame(readings)

    # Ensure columns exist
    if "spo2" not in df.columns or "heart_rate" not in df.columns:
        print("❌ Missing required columns `spo2` or `heart_rate`")
        return

    # Assign time column (5 sec interval)
    df["time_sec"] = np.arange(0, len(df) * 5, 5)

    # Make time_sec first column
    cols = ["time_sec"] + [col for col in df.columns if col != "time_sec"]
    df = df[cols]

    df = detect_apnea(df)

    print("\n📥 Updated Data with Apnea Labels:")
    print(df.tail())

    csv_file = f"patient_{patient_id}_data.csv"
    df.to_csv(csv_file, index=False)

    print(f"\n🩺 CSV Updated: {csv_file}  (0=Normal, 1=Possible-Apnea)")

# ---------------------------------------------------
# Start stream
# ---------------------------------------------------
print(f"🔍 Listening for real-time updates at: {data_path}")
my_stream = db.child(data_path).stream(stream_handler)

while True:
    pass
