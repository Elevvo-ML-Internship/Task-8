# 🚦 Traffic Sign Recognition (CNN - TensorFlow/Keras)

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** to classify **43 different traffic sign classes** using the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).  
The model is trained on labeled images and achieves high accuracy on unseen test data.

---

## 🛠️ Tech Stack

- **Python**    
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib, Seaborn**
- **PIL (image preprocessing)**
- **Scikit-learn (confusion matrix, classification report)**

---

## 📂 Dataset


### 1) Download (Kaggle CLI)

```
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p dataset
cd dataset
unzip gtsrb-german-traffic-sign.zip
cd ..
```

### 2) Manual Download

- Go to the [dataset link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Download the ZIP file and extract it inside a folder called `dataset/`

### 3) Expected Structure

```
project/
├─ dataset/
│  ├─ Train.csv
│  ├─ Test.csv
│  ├─ Train/    # images in subfolders (0..42)
│  └─ Test/     # test images
├─ train.py or notebook.ipynb
└─ README.md
```

- **Train.csv** → Training image paths & labels    
- **Test.csv** → Test image paths & labels
- **43 Classes** (0–42), each representing a unique traffic sign.

Each image is resized to **64×64** and normalized before feeding into the CNN.

---

## 🏗️ Model Architecture

- **Conv2D(32, 3×3, ReLU) + MaxPooling2D**    
- **Conv2D(64, 3×3, ReLU) + MaxPooling2D**
- **Flatten**
- **Dense(128, ReLU) + Dropout(0.5)**
- **Dense(43, Softmax)**

Optimizer: **Adam**  
Loss: **Categorical Crossentropy**  
Callbacks: **EarlyStopping (patience=3, restore best weights)**

---

## 📊 Results

### 🔹 Training Performance

- **Final Train Accuracy**: ~98.7%    
- **Final Val Accuracy**: ~99.4%

PIC

---
### 🔹 Confusion Matrix

The model performs well across all classes with very few misclassifications.

PIC

---

### 🔹 Test Performance

- **Test Accuracy**: **97.13%** (12,630 samples)
- **Classification Report:**
	- **Macro Avg F1-score**: 0.96
	- **Weighted Avg F1-score**: 0.97

---
## 📌 Conclusion

The CNN achieves **state-of-the-art performance (~97% test accuracy)** on the traffic sign dataset.  
With further augmentation and deeper architectures, accuracy can be pushed beyond **99%**, making this approach highly effective for **real-world autonomous driving systems**.