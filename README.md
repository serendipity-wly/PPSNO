# PPSNO
An Efficient and Feature-rich SNO Sites Predictor Through Stacking Ensemble Learning Strategy from Protein Sequence-derived Information
# First, create a virtual environment using conda.
conda create -n name python==3.7
# Then, install the necessary packages on the created virtual environment
conda install tensorflow==2.6.0
conda install tensorflow-gpu==2.6.0
conda install transformers==3.4.0
conda install xgboost==1.5.2
conda install catboost==0.26.1
conda install matplotlib==3.3.4
conda install pandas==1.1.5
conda install scikit-learn==0.24.2
# Feature extraction
Run a Python program for feature extraction, for example, the PSSM feature extraction method, by executing the 'PSSM.py' file.
# Note
The PSSM files required for the PSSM feature extraction method have been pre-processed and stored in their respective directories.They can be directly read and utilized.You can merge all the results from different feature methods and save them into a single text file for easy access in subsequent steps.
# Model
