# Pneumonia Detection using Chest X-ray Images

Pneumonia is an inflammatory condition primarily affecting the lungs, characterized by symptoms such as cough, chest pain, fever, and difficulty breathing. The goal of this project is to develop an automated system for detecting and classifying pneumonia in medical images.

![Symptoms of Pneumonia](https://user-images.githubusercontent.com/65142149/215302250-841fde71-e182-4ffd-8036-625a3a717de7.png)

## Motivation
The motivation behind this project is to leverage artificial intelligence to accurately detect and classify pneumonia in humans using chest X-ray images. By automating the diagnosis process, it can aid healthcare professionals in providing timely and accurate treatment.

## Approach
A Convolutional Neural Network (CNN) architecture was built to distinguish between normal and pneumonia lungs. 

### Fixing the Class Imbalance Bias
The original Kaggle dataset contains highly imbalanced classes (significantly more Pneumonia samples than Normal). When initially trained, the model exhibited a severe bias and over-predicted Pneumonia, leading to a high false-positive rate. 

To resolve this, **Class Weighting** was introduced in the training pipeline (`train.py`). The model was re-trained while heavily penalizing mistakes on the minority class (Normal). This successfully removed the bias and achieved:
- **Normal Recall: ~82%** (improved from ~0%)
- **Pneumonia Recall: ~99%**
- **AUC-ROC: 0.9685**

### Web Application with Decision Threshold
The system includes a local web application built with **Streamlit**. It features a configurable **Confidence Threshold Slider**, allowing healthcare professionals to adjust the required confidence level for diagnosing Pneumonia.

## Key Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- Scikit-Learn
- OpenCV & Pillow

## How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/KoustavaRamesha/federated-learning-model.git
cd federated-learning-model
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
pip install streamlit scikit-learn
```

3. **Run the Streamlit Application:**
```bash
streamlit run xray_web.py
```
This will launch the web application in your browser (usually at `http://localhost:8501`), where you can upload Chest X-Ray images and adjust the decision threshold to get a diagnosis.

## The Dataset
The dataset used in this project is organized into three folders: train, test, and val. It consists of X-ray images (JPEG) categorized into two classes: Pneumonia and Normal.

Chest X-ray images (anterior-posterior) were obtained from pediatric patients between the ages of one to five years old from the Guangzhou Women and Children’s Medical Center. 

The dataset is available on Kaggle: [Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)

## Achievement
The developed system serves as a prototype for potential application in the field of biomedical imaging, providing a valuable tool for diagnosing pneumonia accurately and efficiently with minimized false-positive bias.

**Credit:** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
