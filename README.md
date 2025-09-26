# Plant Doctor

Plant Doctor is a deep learning-powered application designed to help identify plant diseases from leaf images.
It uses a **ResNet18-based CNN model** trained on the [PlantVillage dataset](https://www.tensorflow.org/datasets/catalog/plant_village) and [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset), making it a valuable tool for farmers, researchers, and anyone interested in precision agriculture.

---

## Features

* Detects multiple plant diseases from leaf images with high accuracy.
* Powered by **PyTorch** and **ResNet18** architecture.
* Easy-to-use **UI for predictions**.
* A **Gallery** of Plant Leaf Images
* Extensible training pipeline for experimenting with new models.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Debottam1234567890/Plant-Doctor.git
cd Plant_Disease_Detection
pip install -r requirements.txt
```

---

## Usage

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
Note: The actual hosted app (plant-doctor-ai.streamlit.app) has to be "waken up" at times due to streamlit requirements, but the actual loading of the app takes 2-3 seconds only.
---

## Dataset

This project uses the **PlantVillage dataset** and **PlantDoc dataset**, which contain labeled images of healthy and diseased leaves across multiple crops.

> The dataset and model file is not included in this repository due to size constraints. The model file included here is for demonstration purposes only. The user has to train the model on the two aforementioned datasets using engine.py which will create the model file used for predictions.
> You can download the PlantVillage dataset from [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).
> You can download the PlantDoc dataset from [PlantDoc on GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)

---

## Future Plans

* Expand to support **ResNet50/ResNet101** for better accuracy.
* Optimize model for **mobile deployment**.
* Potential integration with a **startup solution** for farmers.

---

## License

This project is licensed under the **Apache 2.0 License** – free to use, modify, and distribute with attribution.

---

## Acknowledgements

* [PlantVillage Dataset](https://plantvillage.psu.edu/)
* [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
* [PyTorch](https://pytorch.org/)
* [ResNet Architecture](https://arxiv.org/abs/1512.03385)

---

## About

This project is part of my journey in **deep learning, computer vision, and AI for agriculture**.
Long-term vision: **democratize AI tools for farmers** to improve yield and reduce crop losses globally.
