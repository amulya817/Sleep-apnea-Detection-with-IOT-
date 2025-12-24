# 💤 Sleep Apnea Detection Using Machine Learning and IoT

An IoT-enabled wearable system for real-time sleep apnea detection using physiological signals (Heart Rate and SpO₂), cloud monitoring, and machine learning classification.

---

## 📌 Problem Statement
Sleep apnea is a widely underdiagnosed sleep disorder where breathing repeatedly stops during sleep, increasing the risk of cardiovascular disease, stroke, and reduced quality of life. Traditional diagnosis relies on expensive, hospital-based polysomnography (PSG) and lacks continuous home monitoring.

---

## ✅ Proposed Solution
This project proposes a compact wearable system that continuously monitors Heart Rate (HR) and Oxygen Saturation (SpO₂) using IoT sensors. The collected data is transmitted to the cloud and analyzed using machine learning to detect apnea events in real time.

---

## 🧠 Key Features
- Wearable IoT-based sleep monitoring device
- Continuous Heart Rate and SpO₂ monitoring
- Real-time data upload to Firebase Cloud
- Machine Learning–based apnea detection
- Interactive Streamlit web dashboard
- Email alert notifications
- GSM backup for network failure

---

## 🏗️ System Architecture
1. MAX30102 sensor captures HR and SpO₂ data  
2. ESP32-C3 processes and transmits data  
3. Firebase Realtime Database stores sensor data  
4. Machine Learning model classifies Normal vs Apnea  
5. Streamlit dashboard visualizes results and sends alerts  

---

## 🔬 Methodology

### 1. Data Collection
- Pre-recorded physiological dataset (HR & SpO₂)
- Real-time data from wearable IoT device

### 2. Data Preprocessing
- Noise removal and signal stabilization  
- Handling missing values  
- Normalization of signals  
- Time-window segmentation  

### 3. Feature Engineering
- SpO₂ desaturation events (< 90%)  
- Heart rate variability  
- Time-based physiological variations  

### 4. Model Training and Evaluation

| Model | Accuracy | Remarks |
|------|---------|--------|
| Naive Bayes | 75.96% | Poor apnea detection |
| Random Forest | **97%** | Best performance |

SMOTE was applied to handle class imbalance.

---

## 🧪 Results
- Random Forest achieved 97% accuracy
- Improved detection of apnea events
- Stable performance on real-time data
- Effective visualization using dashboard

---

## 🧰 Hardware Components
- ESP32-C3 Microcontroller  
- MAX30102 Heart Rate & SpO₂ Sensor  
- OLED Display (1.2 inch)  
- SIM800L GSM Module  
- Lithium Polymer (Li-Po) Battery  

---

## 🖥️ Software Stack
- Python
- Scikit-learn
- Firebase Realtime Database
- Streamlit
- Embedded C (ESP32)
- SMOTE (imbalanced-learn)

---

## 📊 Web Dashboard
- Real-time patient data visualization
- Apnea prediction results
- Report generation
- Email alert notifications

---

## 🎯 Conclusion
This project demonstrates an end-to-end intelligent sleep apnea detection system integrating IoT, Machine Learning, and Cloud Computing. The system achieves high accuracy and supports real-time monitoring, making it suitable for affordable home-based sleep health assessment.

---

## 👩‍💻 Authors
- Amulya B  
- Nivasini B  
- Saniya Sulthana  
- Yashaswini R K  

Department of Artificial Intelligence & Machine Learning

---

## 📚 References
Refer to the project documentation for the complete list of references.
