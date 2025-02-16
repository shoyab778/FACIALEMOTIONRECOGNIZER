# Facial_Emotion_Recognizer_Model
Real-Time Emotion Detection System

A real-time emotion detection system using OpenCV, TensorFlow, and Keras, which captures facial expressions from a webcam and classifies them into different emotions.

📌 Features
- Real-Time Emotion Recognition using a deep learning model.
- Face Detection with OpenCV’s Haar Cascade Classifier.
- Live Webcam Support to capture facial expressions.
- Fast and Efficient Predictions using a trained deep learning model.
- User-Friendly Interface displaying detected emotions.

🛠️ Technologies Used
- Python for scripting.
- OpenCV for image and video processing.
- TensorFlow/Keras for deep learning-based emotion detection.
- NumPy for numerical operations.
- Haar Cascade Classifier for face detection.

🔑 Prerequisites
Ensure the following dependencies are installed before running the project:
```bash
pip install opencv-python numpy tensorflow keras
```

📥 Installation & Usage
1. Clone or download the repository.
2. Ensure the trained model (`facialemotionmodel.h5`) is available at the specified path.
3. Run the script using:
   ```bash
   python realtimedetection.py
   ```
4. Press 'q' to exit the webcam feed.

📚 Training & Testing Steps
The `trainmodel.ipynb` file contains the training process:
1. Dataset Preprocessing – Load and prepare emotion-based datasets.
2. Model Architecture – Build a Convolutional Neural Network (CNN).
3. Training the Model – Train the model using Keras & TensorFlow.
4. Evaluation – Validate performance using accuracy metrics.
5. Save the Model – Export it as `facialemotionmodel.h5`.

📂 Code Structure
- realtimedetection.py – Main script for real-time emotion detection.
- trainmodel.ipynb – Jupyter notebook for training the model.
- facialemotionmodel.h5 – Pre-trained model for inference.

🖥️ API Usage
The model is loaded using:
```python
model = keras.models.load_model("path_to_model.h5")
```
Prediction is performed on a detected face:
```python
prediction = model.predict(roi_gray)
emotion = emotion_labels[np.argmax(prediction)]
```

⚠️ Error Handling
- Model Loading Issues: Checks if the model file exists.
- Webcam Issues: Handles cases where the camera is unavailable.
- Face Detection Failures: Ensures robust detection in different lighting.

🚀 Future Improvements
- Improve model accuracy with larger datasets.
- Optimize performance using TensorFlow Lite.
- Add support for multiple faces in a frame.
- Integrate with a mobile app for wider usability.

💡 Contributing
- Fork the repository.
- Create a feature branch.
- Submit a pull request for review.

📧 For queries, contact: smdshoyab07@gmail.com


