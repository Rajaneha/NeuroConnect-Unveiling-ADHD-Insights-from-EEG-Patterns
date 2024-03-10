**EEG data analysis and modelling**
# ADHD EEG Data Analysis and Interface

## Overview

This project analyzes EEG data from the ADHD dataset obtained from the IEEE data portal. The dataset comprises EEG data for 121 subjects, with 60 subjects having ADHD and 61 serving as controls. The preprocessing involved converting MAT files to CSV, normalizing row counts, and calculating nonlinear correlation coefficients to generate 19x19 matrices for each subject.

## Analysis Steps

1. **Data Preprocessing:**
    - Convert MAT files to CSV.
    - Normalize row counts.

2. **Nonlinear Correlation Analysis:**
    - Compute a 19x19 matrix for each subject by finding the nonlinear correlation coefficient for every column with every other column.

3. **Visualization:**
    - Plot topoplots for one subject from ADHD and one from the control group for inference.

4. **Matrix Processing:**
    - Calculate median values for each cell, resulting in one matrix for each class (ADHD and Control).

5. **Spectral Clustering:**
    - Apply spectral clustering on matrices to cluster electrodes.
    - Apply spectral clustering based on lobes and hemispheres.

6. **Rank Analysis:**
    - Find the rank of matrices and infer the values.

## Folder Structure

- `ADHD`: Contains CSV files for subjects with ADHD.
- `Control`: Contains CSV files for control subjects.
- `EEG_Analysis_with_interface.py`: Code including Streamlit for user interface representation.
- `EEG_Analysis.py`: Code with analysis and machine learning without UI.

## How to Use

1. Navigate to the `EEG_Analysis_with_interface.py` file.
2. Run the script to launch the Streamlit interface.

Feel free to explore the folders for additional details on the dataset, code, and results.


