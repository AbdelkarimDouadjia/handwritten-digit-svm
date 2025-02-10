# Handwritten Digit Recognition with SVM

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Decision Boundary Visualization](#decision-boundary-visualization)
  - [Predictions on New Data](#predictions-on-new-data)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Google Colab Notebook](#google-colab-notebook)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview
This project focuses on classifying images of handwritten digits into their respective numerical values (0–9) using **Support Vector Machines (SVM)**. Leveraging the **Digits dataset** from **scikit-learn**, the project implements both a custom SVM and scikit-learn's SVM with an RBF kernel. The aim is to achieve high accuracy in digit recognition through a well-defined machine learning pipeline.

## Dataset
The project utilizes the **Digits dataset** from scikit-learn, characterized by:
- **Shape:** (1797, 64) — Each sample is an 8×8 grayscale image flattened into a 64-dimensional vector.
- **Classes:** 10 (digits 0 through 9)
- **Normalization:** Pixel values are scaled to the range [0, 1] for uniformity.
- **Train-Test Split:** The data is divided into 80% training and 20% testing sets.

## Project Files
This repository includes the following files:
- `3_SVM.ipynb`: Jupyter Notebook containing code for data exploration, preprocessing, model training, evaluation, and visualization.
- `3_SVM_report.pdf`: A detailed report summarizing the project's methodology, experiments, and performance metrics.
- `03. SVM.pdf`: Instructions outlining the tasks, methodology, and guidelines for the project.

## Methodology

### Data Exploration and Preprocessing
- **Data Loading:** The Digits dataset is loaded directly from scikit-learn.
- **Exploration:** Initial analysis is performed to understand the data structure and distribution.
- **Normalization:** Pixel values are normalized to the [0, 1] range.
- **Train-Test Split:** The dataset is split into 80% training and 20% testing data.

### Model Training
- **Custom SVM Implementation:** An SVM model is implemented from scratch based on classroom concepts.
- **Scikit-learn SVM:** An SVM using an RBF kernel is employed via scikit-learn to capture non-linear relationships.

### Model Evaluation
- **Performance Metrics:** Evaluation of the SVM model includes:
  - **Accuracy:** Achieved 98% accuracy.
  - **F1 Score:** Achieved an F1 score of 98%.
- **Confusion Matrix:** Visualization to analyze the distribution of correct versus incorrect predictions.

### Decision Boundary Visualization
- **PCA Reduction:** Principal Component Analysis (PCA) is applied to reduce dimensionality for visualization purposes.
- **Visualization:** The decision boundaries are plotted to illustrate how the SVM differentiates between digit classes.

### Predictions on New Data
- **New Image Testing:** The trained model is used to predict labels for new, unseen handwritten digit images.
- **Deployment Considerations:** Discusses challenges such as image variations and noise that may impact real-world performance.

## Results and Insights
- **High Accuracy:** The SVM model demonstrated near-perfect accuracy (98%) on the test set.
- **Effective Classification:** The confusion matrix shows that most predictions lie along the diagonal, indicating robust classification.
- **Visualization Benefits:** PCA-based plots reveal clear separations between most digit classes, with slight overlaps for visually similar digits.

## Limitations and Future Improvements
- **Dataset Constraints:** The Digits dataset is relatively small, which might limit the model's generalization to more varied handwriting.
- **Model Generalization:** Future work could incorporate data augmentation and more extensive hyperparameter tuning.
- **Advanced Techniques:** Exploring deep learning methods such as Convolutional Neural Networks (CNNs) could further enhance performance.

## Technologies Used
- **Python**
- **Scikit-learn:** For SVM implementation and dataset handling.
- **NumPy & Pandas:** For data processing and manipulation.
- **Matplotlib & Seaborn:** For data visualization.
- **Jupyter Notebook / Google Colab:** For interactive development and experimentation.

## How to Run the Project Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/handwritten-digit-svm.git
   cd handwritten-digit-svm
   ```
2. **Install the Required Libraries:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```
3. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the `3_SVM.ipynb` notebook and run the cells sequentially.

## Google Colab Notebook
Alternatively, you can run the project on Google Colab using the following link:
[Google Colab Notebook](https://colab.research.google.com/drive/1Jw1Bk67cDD4z_bmjRFigBXxcWQ3ltiIM?usp=sharing)

## Contributors
- **Douadjia Abdelkarim**  
  Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- **Scikit-learn:** For providing the Digits dataset and SVM tools.
- **University Instructors:** For guidance on implementing SVM.
- **Djilali Bounaama University:** For academic support and resources.

---
This project is part of coursework on **Machine Learning with Support Vector Machines (SVM)** and aims to provide practical experience in image classification.
