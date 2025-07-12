# 👁️ Blink2Diag: CNN-Based Blink Detection for Intraocular Pressure Estimation

This project uses computer vision and deep learning techniques to detect eye blinks from facial video data, with the goal of supporting non-invasive intraocular pressure (IOP) estimation. Built with Python, OpenCV, and Keras, the system leverages a Convolutional Neural Network (CNN) trained on labeled eye-region images to classify blink events. Accurate blink detection lays the foundation for developing accessible diagnostic tools for conditions like glaucoma.

---

## 🔬 Background

Traditional methods of estimating IOP can be invasive, expensive, or require in-person clinical resources. Blink frequency and quality, however, have shown potential as indicators for ocular pressure and eye health. This model aims to detect blinks as a preliminary step toward automating parts of the diagnostic process.

---

## 🛠️ Tools & Libraries

- **Python**: Core scripting language  
- **OpenCV**: Real-time video processing and facial landmark extraction  
- **Keras (TensorFlow backend)**: Building and training the CNN model  
- **Dlib & Haar Cascades**: Facial and eye region detection  
- **Git**: Version control and collaboration

---

## 💡 Key Features

- Real-time blink detection using webcam video input  
- Preprocessing pipeline for eye-region extraction  
- CNN model trained on grayscale eye images  
- Live visualization of blink metrics and blink count  
- Ethical considerations to minimize algorithmic bias

---

## ⚖️ Ethics & Impact

Inspired by values from the *Justice and Legal Thought (JLT)* program, this project was developed with ethical awareness around the use of AI in healthcare. We prioritized fairness, representation, and bias mitigation in both dataset handling and model design.

---

## 🚀 Future Work

- Integrate IOP estimation algorithms based on blink data  
- Expand dataset for greater demographic diversity  
- Deploy prototype in clinical or academic settings  

---

> 💬 Have questions or suggestions? Feel free to open an issue or fork the project!

