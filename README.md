🌟 Face Recognition System (OpenCV + Streamlit + LBPH)
<div align="center">










A complete end-to-end Real-Time Face Recognition System with Dataset Creation, Model Training, Live Recognition, and a Streamlit GUI launcher.

</div>
🚀 Features
1️⃣ Dataset Builder (OpenCV GUI)

Captures multiple face images for each user

Auto & Manual capture modes

Ensures stable face tracking

Stores images neatly under dataset/<person-name>/

2️⃣ Model Trainer

Reads labeled face folders

Preprocesses images

Trains LBPH (Local Binary Patterns Histogram) recognizer

Saves:
✔ lbph_model.yml
✔ labels.pickle

3️⃣ Live Recognition

Real-time face detection & prediction

Shows label + confidence

Marks unknown faces

Press 'a' to add new person directly from camera

Automatically retrains and reloads the model

4️⃣ Streamlit GUI (Main App)

Clean and simple UI with 3 tabs:

Capture Dataset

Train Model

Live Recognition

Runs all modules without terminal commands

Snapshot saving option

Works with any connected webcam

📁 Project Structure
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

🔧 Installation
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd Face-Recognition-App

2. Install dependencies
pip install -r requirements.txt


⚠️ Make sure you installed opencv-contrib-python, not the regular OpenCV.

▶️ Usage
Start the GUI
streamlit run app.py


From the GUI, you can:

📸 Capture Dataset

Enter the person’s name

Capture images automatically or manually

🧠 Train Model

Reads all dataset folders

Rebuilds the LBPH classifier

👁 Live Recognition

Detects & identifies faces

Shows confidence

Press 'a' to add a new person on the fly

Add your demo images or GIFs in an assets/ folder.

![Dataset Builder](assets/dataset.png)
![Recognition Demo](assets/recognition.gif)

⚙️ Requirements

Python 3.8+

OpenCV (contrib version)

Streamlit

Numpy

Install everything via:

pip install -r requirements.txt

📌 Why LBPH?

LBPH is:

Fast

Lightweight

Works without GPU

Great for small datasets

Easy retraining

Perfect for local face recognition apps.

🚧 Future Improvements

Replace LBPH → FaceNet / ArcFace (Deep Learning)

Add Face Enrollment Form inside Streamlit

Store user data in a database

Add attendance system

Deploy on Streamlit Cloud

Add logging & analytics

❤️ Contributing

Contributions, issues, and feature requests are welcome!

How to contribute:

Fork the repo

Create a branch: git checkout -b feature-new

Commit changes

Create a pull request

📝 License

This project is released under the MIT License.

⭐ Show Your Support

If you find this project useful:

👉 Star the repo on GitHub
👉 Fork it and build your own version

🙌 About This Project

This repo demonstrates a full real-time Face Recognition workflow designed for:

Students

ML beginners

AI portfolio building

Security automation demos

Attendance system prototypes
