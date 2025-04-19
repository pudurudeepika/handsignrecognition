# ✋ Real-Time Hand Sign Recognition

This project is about recognizing hand signs in real-time using a webcam and a deep learning model. It detects American Sign Language (ASL) letters (except J and Z) and shows them as text on the screen.

## 🔍 What It Does

- Captures live video from a webcam.
- Detects the hand using **MediaPipe**.
- Picks the hand area (Region of Interest).
- Prepares the image for prediction.
- Uses a **CNN model** to predict the hand sign.
- Shows the letter on the screen and forms a sentence.

## 🧰 Tools and Technologies

- **Python**
- **OpenCV** – For capturing webcam video.
- **MediaPipe** – For detecting hand landmarks.
- **TensorFlow / Keras** – For training and using the deep learning model.
- **CNN (Convolutional Neural Network)** – For recognizing hand signs from images.

## 📐 How It Works

1. Capture a frame from the webcam.
2. Detect hand landmarks using MediaPipe.
3. Extract and resize the hand image to 64x64 pixels.
4. Use the CNN model to predict the sign.
5. Show the result in real-time.

## 🧠 Model Info

- Input: 64x64 RGB image
- Output: 24 letters (A–Y, except J and Z)
- Trained on a dataset of ASL hand signs

  ## 📂 Dataset

The dataset used for training the CNN model can be downloaded from the link below:

🔗 [Download Dataset (Kaggle)](https://www.kaggle.com/datasets/ash2703/handsignimages)


## 📸 Example Output

The system can detect and form words like “CALL” by recognizing each letter one by one from hand gestures.

![Screenshot 2025-04-12 114833](https://github.com/user-attachments/assets/8c2a0034-b61e-442e-8f54-acad1b20cbb6)

## ⚠️ Problems Faced

- In the beginning, the model was not accurate because of a small dataset.
- The dataset did not have letters J and Z because they involve motion.

## ✅ Improvements Made

- Used a bigger and better dataset with more variety.

- Improved the model’s accuracy and performance.

## 📸 Demo

> Below is an example prediction of the word **"CALL"** formed by recognizing individual hand signs:
> 
![Screenshot 2025-04-12 114618](https://github.com/user-attachments/assets/c6d2fe66-8f32-4246-abcd-0040f831254e)
![Screenshot 2025-04-12 114637](https://github.com/user-attachments/assets/dae0fc0b-de1a-479d-8f2a-5fab463fa38a)
![Screenshot 2025-04-12 114821](https://github.com/user-attachments/assets/c81bd78d-765a-4a43-98bb-5a911e07af57)
![Screenshot 2025-04-12 114833](https://github.com/user-attachments/assets/8c2a0034-b61e-442e-8f54-acad1b20cbb6)
