# DS4002 Case Study: Fruit Image Classification Challenge

Welcome! This repository contains all the materials you'll need to embark on an exciting case study: your mission is to explore a dataset of fruit images, and then design, build, train, and evaluate a Convolutional Neural Network (CNN) to classify these fruits.

This project will give you hands-on experience with a real-world image classification task, from understanding the data to interpreting a deep learning model's performance.

## Learning Objectives

By completing this case study, you will:
*   Gain a practical understanding of the image classification workflow.
*   Learn how to perform Exploratory Data Analysis (EDA) on an image dataset.
*   Understand the fundamentals of Convolutional Neural Networks (CNNs) and the concept of transfer learning (e.g., using a pre-trained model like ResNet50 as a base).
*   Develop skills in building and training a deep learning model using Python libraries like TensorFlow/Keras.
*   Learn to evaluate model performance using metrics like accuracy, precision, recall, and confusion matrices.
*   Practice documenting and presenting your findings in a technical report.

## Repository Contents

Here's a breakdown of what you'll find in this repository:

*   **`CS3 Tamer El Aridi Hook.pdf`**: This is your main project prompt! It sets the scene and describes the overall challenge. **Start by reading this.**
*   **`Case Study Rubric.pdf`**: This document details the expectations for your submission, including the deliverables and how your work will be assessed. **Read this carefully to understand what's required.**
*   **`README.md`**: This file you're currently reading!
*   **`SCRIPTS/`**:
    *   **`1_Fruit_EDA.ipynb`**: A Jupyter Notebook to guide you through the Exploratory Data Analysis of the fruit dataset. You will run this notebook to understand the data's characteristics.
*   **`DATA/`**:
    *   **`Fruits_Dataset_Train/`**: Contains all the training image folders, organized by class.
    *   **`Fruits_Dataset_Test/`**: Contains all the testing image folders, organized by class.
    *   **`Labels_Train.csv`**: CSV file containing the filenames and corresponding one-hot encoded labels for the training images.
    *   **`Labels_Test.csv`**: CSV file containing the filenames and corresponding one-hot encoded labels for the testing images.
    *   **`Fruit_Summary.jpg`**: A visual summary/collage of the fruits in the dataset.
    *   **`Sample_Images/`**: A small collection of sample images from the dataset.
*   **`REFERENCES/`**:
    *   **`The Basics of ResNet50 _ ml-articles – Weights & Biases.pdf`**: An article explaining the ResNet50 architecture, which you might consider using for transfer learning.
    *   **`Understanding Transfer Learning for Deep Learning.pdf`**: An article to help you understand the concept of transfer learning.

## Prerequisites & Setup

1.  **Python Environment**: Python 3.7+ is recommended.
2.  **Key Libraries**: You'll need libraries such as:
    *   `tensorflow` (for building and training your neural network)
    *   `keras` (often comes with TensorFlow)
    *   `pandas` (for handling CSV label files)
    *   `matplotlib` & `seaborn` (for plotting during EDA and results visualization)
    *   `scikit-learn` (for metrics like confusion matrix, classification report)
    *   `Pillow` (PIL) (for image manipulation, used in EDA)
    *   `numpy`
3.  **Recommended Platform**: **Google Colab** is highly recommended, especially for training your model, as it provides free GPU access which significantly speeds up the training process.
    *   If using Google Colab, you can upload this entire repository (or clone it directly into your Colab environment) to easily access all files.

## How to Get Started: Your Case Study Workflow

1.  **Familiarize Yourself (Crucial First Steps):**
    *   Thoroughly read the **`CS3 Tamer El Aridi Hook.pdf`** to understand the case study's scenario and objectives.
    *   Carefully review the **`Case Study Rubric.pdf`** to understand all requirements, deliverables, and grading criteria.
2.  **Set Up Your Environment:**
    *   Clone or download this repository.
    *   Set up your Python environment locally or prepare to use Google Colab. All necessary data is included in the `DATA/` folder.
3.  **Exploratory Data Analysis (EDA):**
    *   Open and run the `SCRIPTS/1_Fruit_EDA.ipynb` notebook.
    *   Follow the instructions within the notebook, execute the cells, and analyze the outputs to understand the dataset.
    *   Make notes of your key observations; these will be useful for your report and for designing your model.
4.  **Model Development (Your Core Task!):**
    *   **Create a new Jupyter Notebook** in the `SCRIPTS/` folder (e.g., name it `2_Fruit_Classifier_CNN.ipynb`). This notebook will be where you:
        *   Load and preprocess the image data from the `DATA/` folder (resizing, normalization, creating data generators/pipelines).
        *   Define your Convolutional Neural Network (CNN) architecture. Consider using transfer learning with a pre-trained model like ResNet50 (see `REFERENCES/`) as a starting point.
        *   Compile your model (choose an optimizer, loss function, and metrics).
        *   Train your model on the training data, using the testing data (or a validation split from training data) to monitor performance.
        *   Evaluate your trained model on the unseen test set.
        *   Generate visualizations of your results (e.g., training history plots, confusion matrix).
    *   **Guidance:** Refer to the concepts outlined in the `CS3 Tamer El Aridi Hook.pdf` and the articles in the `REFERENCES/` folder to help you design your model and training strategy.
5.  **Report Your Findings:**
    *   Prepare a technical report (PDF document) as specified in the `Case Study Rubric.pdf`.
    *   Your report should summarize your EDA, methodology (preprocessing, model architecture, training), results (performance metrics, visualizations), and a discussion/conclusion.

## Expected Deliverables

Please refer to the **`Case Study Rubric.pdf`** for the definitive list of deliverables and detailed grading criteria. In summary, you will typically submit:
1.  Your completed `1_Fruit_EDA.ipynb` notebook.
2.  Your **created** `2_Fruit_Classifier_CNN.ipynb` (or similarly named) notebook containing your model development, training, and evaluation.
3.  Your final technical report in PDF format.
4.  A link to your version of this GitHub repository containing all your work.

Good luck, and enjoy the challenge of teaching an AI to see!
