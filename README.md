<p align="center">
  <img src="https://img.shields.io/github/stars/Debottam1234567890/Plant-Doctor?style=flat-square" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/Debottam1234567890/Plant-Doctor?style=flat-square" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/license/Debottam1234567890/Plant-Doctor?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/PyTorch-ResNet18-blue?style=flat-square" alt="PyTorch ResNet18"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/streamlit-ui-orange?style=flat-square" alt="Streamlit UI"/>
  <img src="https://img.shields.io/badge/dataset-PlantVillage%20%26%20PlantDoc-green?style=flat-square" alt="Datasets"/>
  <img src="https://img.shields.io/badge/deep%20learning-agriculture-important?style=flat-square" alt="Deep Learning Agriculture"/>
</p>

---

# 🌱 Plant Doctor – Plant Disease Detection

Plant Doctor is a deep learning-powered application designed to help identify plant diseases from leaf images.
It uses a **ResNet18-based CNN model** trained on the [PlantVillage dataset](https://www.tensorflow.org/datasets/catalog/plant_village) and [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset), making it a valuable tool for farmers, researchers, and anyone interested in precision agriculture.

---

## 🚀 Features

* Detects multiple plant diseases from leaf images with high accuracy.
* Powered by **PyTorch** and **ResNet18** architecture.
* Easy-to-use **UI for predictions**.
* A **Gallery** of Plant Leaf Images
* Extensible training pipeline for experimenting with new models.

---

## 📂 Project Structure

```
Plant_Disease_Detection/
│── ui.py                                 # User Interface file
│── engine.py                             # Training & evaluation engine
│── requirements.txt                      # Python dependencies
│── README.md                             # Project documentation
|── resnet18_plant_disease_detection.pth  # Model file
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Debottam1234567890/Plant-Doctor.git
cd Plant_Disease_Detection
pip install -r requirements.txt
```

---

## 🧑‍💻 Usage

### Training

To train the model on your dataset:

```bash
python engine.py
```

### Running Inference (UI)

If you have the UI file (e.g., `ui.py` with Streamlit):

```bash
streamlit run ui.py
```

---

## 🔎 Prediction Example

Below is an example of the Plant Doctor prediction UI and result:

<p align="center">
  <img src="https://raw.githubusercontent.com/Debottam1234567890/Plant-Doctor/main/assets/prediction_screenshot.png" alt="Plant Doctor Prediction Example" width="600"/>
</p>

> *A screenshot of the Streamlit-based Plant Doctor UI, showing an uploaded leaf image and the model's predicted disease result.*

---

## 🎥 Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=D9GXTIfx_KA" target="_blank">
    <img src="https://img.youtube.com/vi/D9GXTIfx_KA/0.jpg" alt="Plant Doctor Demo" width="500"/>
  </a>
</p>

> _Click the image above to watch a walkthrough of Plant Doctor in action!_

---

## 📊 Dataset

This project uses the **PlantVillage dataset** and **PlantDoc dataset**, which contain labeled images of healthy and diseased leaves across multiple crops.

> ⚠️ The dataset and model file is not included in this repository due to size constraints. The model file included here is for demonstration purposes only. The user has to train the model on the two aforementioned datasets using engine.py which will create the model file used for predictions.
> You can download the PlantVillage dataset from [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
> You can download the PlantDoc dataset from [PlantDoc on GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)

---

## 🔮 Future Plans

* Expand to support **ResNet50/ResNet101** for better accuracy.
* Optimize model for **mobile deployment**.
* Potential integration with a **startup solution** for farmers.

---

## 📜 License

This project is licensed under the **Apache 2.0 License** – free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

* [PlantVillage Dataset](https://plantvillage.psu.edu/)
* [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
* [PyTorch](https://pytorch.org/)
* [ResNet Architecture](https://arxiv.org/abs/1512.03385)

---

## 💡 About

This project is part of my journey in **deep learning, computer vision, and AI for agriculture**.
Long-term vision: **democratize AI tools for farmers** to improve yield and reduce crop losses globally.
