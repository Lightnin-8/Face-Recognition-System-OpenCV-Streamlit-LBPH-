# 🌟 Face Recognition System (OpenCV + Streamlit + LBPH)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-LBPH-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-success)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

A complete end-to-end **Real-Time Face Recognition System** with dataset creation, model training, live recognition, and GUI built using Streamlit.

</div>

---

# 🚀 Features

### **1️⃣ Dataset Builder**
- Auto / Manual capture  
- Stable face tracking  
- Saves images under `dataset/<person-name>/`

### **2️⃣ Model Trainer**
- Preprocesses images  
- Trains **LBPH recognizer**  	
- Saves:  
  - `lbph_model.yml`  
  - `labels.pickle`

### **3️⃣ Live Recognition**
- Real-time detection  
- Shows **name + confidence**  
- Handles unknown faces  
- Add new person by pressing **'a'**  
- Auto-retrain functionality  

### **4️⃣ Streamlit GUI**
- 3 Tabs:
  - 📸 Capture Dataset  
  - 🧠 Train Model  
  - 👁 Live Recognition  
- Snapshot saving  
- Full app runs without terminal commands  

---

# 📁 Project Structure
```
Face-Recognition-App/
│
├── app.py                     # Streamlit GUI launcher
├── dataset_builder.py         # Dataset capture tool
├── train_recognizer_fix.py    # LBPH trainer
├── recognize_live.py          # Real-time recognition script
│
├── requirements.txt
├── README.md
│
├── dataset/                   # (auto-created) person folders
│   ├── Alice/
│   ├── Bob/
│
├── lbph_model.yml             # generated model (not included)
└── labels.pickle              # generated labels (not included)
```


---

# 🔧 Installation

### **1. Clone the repository**
```bash
git clone https://github.com/Lightnin-8
/Face-Recognition-System-OpenCV-Streamlit-LBPH-.git
cd Face-Recognition-System-OpenCV-Streamlit-LBPH-
```
```
pip install -r requirements.txt
```
Start the Streamlit App
```
streamlit run app.py
```
GUI Tabs

:camera_flash: Capture Dataset

Enter person name

Capture 40–60 face images

:brain: Train Model

Reads dataset folder

Retrains LBPH classifier

:eye: Live Recognition

Real-time predictions

Shows confidence

Press 'a' → enroll new user live
