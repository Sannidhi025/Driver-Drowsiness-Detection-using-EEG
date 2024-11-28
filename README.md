Driver Drowsiness Detection
**Overview:**
This Python project utilizes an Artificial Neural Network (ANN) to classify EEG data for detecting driver drowsiness. The model processes EEG signals to distinguish between "eyes open" and "eyes closed" states, aiming to identify drowsiness and improve road safety.

**Features:**
EEG Signal Processing:
Bandpass filter for noise removal.
Notch filter to mitigate powerline interference.
Data Preprocessing:
Feature scaling using StandardScaler.
Conversion of datasets into PyTorch tensors for deep learning.
Model Architecture:
A deep ANN with:
5 fully connected layers.
Batch normalization for training stability.
Dropout layers to reduce overfitting.
Sigmoid activation in the final layer for classification.
Training and Testing:
Cross-entropy loss function with class weights to handle class imbalance.
Adam optimizer for efficient convergence.
Metrics: Accuracy, precision, recall, F1-score, AUC-ROC.
Visualization:
Confusion matrix and ROC curve for performance evaluation.

**Dependencies:**
Python Libraries:
torch
torch.nn
scikit-learn
scipy
mne
pandas
matplotlib
seaborn

Setup
Install the required Python libraries using:
bash
Copy code
pip install torch scikit-learn scipy mne pandas matplotlib seaborn

Place the EEG dataset (EEG_Eye_State_Classification.csv) in the specified directory.
Run the script:
bash
Copy code
python ANN_eye_filter.py

**Usage:**
Dataset Input: The script reads EEG data from a CSV file.
Training: Automatically splits data into training (80%) and testing (20%) sets.
Evaluation: Provides model performance metrics and visualizations.
Output
Model accuracy, precision, recall, F1 score, and AUC-ROC.
Confusion matrix heatmap.
ROC curve visualization.
Notes
Ensure the EEG dataset has appropriate signal features and labels for binary classification.
Modify the file paths in the script to match your dataset's location.
This project demonstrates a robust approach to driver drowsiness detection using EEG signals. Further improvements could involve integrating real-time EEG signal acquisition and optimizing the model for deployment in vehicle systems
