****Breast Cancer Detection Using CNN and Attention Mechanism****

This project implements a deep learning model to classify breast cancer images as benign or malignant using the BreakHis dataset.
The architecture is built on ResNet50 with added Squeeze and Excitation Networks (SE-Nets) after each convolutional layer to enhance feature extraction.
The dataset was balanced using SMOTE, and the model was developed using TensorFlow Keras.

**Overview**

Breast cancer detection is a crucial task in medical imaging. This project uses Convolutional Neural Networks (CNN) and Attention Mechanisms to improve classification accuracy by focusing on important image features.

**Key Features**

Binary Classification: _Classifies images as either benign or malignant._
ResNet50: _Uses ResNet50 as the base model for feature extraction._
Squeeze and Excitation Networks: _Enhances feature representation at each layer._
Dataset Balancing: _Uses SMOTE to handle class imbalance in the BreakHis dataset._
Framework: _Built using TensorFlow Keras for model development._

**Dataset**

BreakHis Dataset: _7,909 images with magnifications at 40x, 100x, 200x, and 400x._
Benign Images: _2,480_
Malignant Images: _5,429_
Image format: _.PNG_

Data augmentation and SMOTE were applied to balance the dataset and improve model robustness.

**Model Architecture**

Base Model: _Pretrained ResNet50 (ImageNet)._
Attention Mechanism: _SE-Nets after each ResNet50 convolutional block._
Optimizer: _Adam with a learning rate of 0.0001._
Loss Function: _Binary cross-entropy._

****System Requirements****

**Minimum Hardware:**

Processor: _Intel Core i3 or equivalent_
Memory: _8 GB RAM (16 GB recommended for larger datasets)_
Storage: _At least 20 GB free space for dataset and models_
**Minimum Software:**
Operating System: _Windows, macOS, or Linux (Ubuntu 20.04+ recommended)_
Python Version: _Python 3.8+_
Libraries:
_TensorFlow 2.12+
Keras
Scikit-learn (for SMOTE)
NumPy
Matplotlib
OpenCV_
**Optional:**
GPU: _CUDA-enabled GPU with at least 4 GB VRAM (for faster training)_

**Installation**

Clone the repository:

git clone https://github.com/jacob-kevin/reast-Cancer-Detection-using-CNN-and-Attention-Mechanisms.git
cd reast-Cancer-Detection-using-CNN-and-Attention-Mechanisms


**Install the required dependencies:**

pip install -r requirements.txt


**Usage**

Open the Jupyter Notebook SE_ResNet50v1.ipynb.

Run the cells step by step to:

Preprocess the data (including augmentation and SMOTE balancing).
Build and train the CNN model (with ResNet50 and SE-Nets).
Evaluate the model's performance on the test set.
Results
The model was trained for 10 epochs and achieved the following performance:

Accuracy: 98.29%
Precision: 99.79%
Recall: 96.71%
F1 Score: 98.22%

**Files in the Repository** 

README.md: Project overview.

requirements.txt: _List of dependencies (TensorFlow, SMOTE, etc.)._

ResNet50.ipynb: _A Jupyter notebook that demonstrates the results obtained using only the ResNet50 model without the attention mechanism._

SE_ResNet50v1.ipynb: _Main Jupyter Notebook containing the full project (preprocessing, model training, and evaluation)._
