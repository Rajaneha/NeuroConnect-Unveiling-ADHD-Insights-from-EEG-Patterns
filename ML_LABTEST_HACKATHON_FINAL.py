#!/usr/bin/env python
# coding: utf-8

# # DATA PREPROCESSING

# In[196]:


import os
import pandas as pd
import numpy as np


def normalize_dataframe(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)

        max_allowed_value = 1e10  
        df[column] = np.clip(df[column], -max_allowed_value, max_allowed_value)
        
        mean = df[column].mean()
        std = df[column].std()
        
        df[column] = (df[column] - mean) / std


def normalize_csv(input_csv_path):
    df = pd.read_csv(input_csv_path)
    normalize_dataframe(df)
    df.to_csv(input_csv_path, index=False) 


folder1_path = 'D:/Sem5/ML LAB TEST/CSV/ADHD'
folder2_path = 'D:/Sem5/ML LAB TEST/CSV/Control part'


for folder_path in [folder1_path, folder2_path]:
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(folder_path, filename)
            normalize_csv(input_csv_path)


# In[197]:


import os
import pandas as pd


def find_min_rows_in_folder(folder_path):
    min_rows = float('inf')
    min_csv_file = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)
            if num_rows < min_rows:
                min_rows = num_rows
                min_csv_file = file_path

    return min_csv_file, min_rows


def normalize_csv(input_csv_path, target_num_rows):
    df = pd.read_csv(input_csv_path)
    df = df.iloc[:, 1:]
    normalized_df = df.sample(target_num_rows, replace=True)
    normalized_df.to_csv(input_csv_path, index=False) 

folder1_path = 'D:/Sem5/ML LAB TEST/CSV/ADHD'
folder2_path = 'D:/Sem5/ML LAB TEST/CSV/Control part'


min_csv_file1, min_rows1 = find_min_rows_in_folder(folder1_path)
min_csv_file2, min_rows2 = find_min_rows_in_folder(folder2_path)


if min_rows1 < min_rows2:
    min_csv_file = min_csv_file1
    min_rows = min_rows1
else:
    min_csv_file = min_csv_file2
    min_rows = min_rows2


for folder_path in [folder1_path, folder2_path]:
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(folder_path, filename)
            normalize_csv(input_csv_path, min_rows)

print(f"Min number of rows {min_rows}")


# In[282]:


import os
import pandas as pd

folder_path = 'D:/Sem5/ML LAB TEST/CSV'

csv_shapes = {}


for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        csv_shapes[filename] = df.shape
        
for filename, shape in csv_shapes.items():
    print(f'{filename}: {shape}')


# # NON-LINEAR CORRELATION CO-EFFICIENT

# In[165]:


from scipy.stats import spearmanr, kendalltau


# In[199]:


import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


def calculate_nonlinear_correlations(input_csv_path):
    df = pd.read_csv(input_csv_path)
    features = df.columns[:19]
    df_subset = df[features]
    
    spearman_corr = df_subset.corr(method='spearman')
    kendall_corr = df_subset.corr(method='kendall')
    return spearman_corr, kendall_corr


def calculate_nonlinear_correlations_in_folder(folder_path, min_rows):
    spearman_correlations = {}
    kendall_correlations = {}
    all_spearman_matrices = []  
    all_kendall_matrices = []   
    count_spearman = 0
    count_kendall = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(folder_path, filename)
            spearman_corr, kendall_corr = calculate_nonlinear_correlations(input_csv_path)
            
            all_spearman_matrices.append(spearman_corr)
            all_kendall_matrices.append(kendall_corr)
    
            spearman_correlations[filename] = spearman_corr
            kendall_correlations[filename] = kendall_corr
            
            count_spearman += 1
            count_kendall += 1

    return (
        spearman_correlations, kendall_correlations,
        all_spearman_matrices, all_kendall_matrices,
        count_spearman, count_kendall
    )

(spearman_correlations_folder1, kendall_correlations_folder1,
 all_spearman_matrices_folder1, all_kendall_matrices_folder1,
 count_spearman1, count_kendall1) = calculate_nonlinear_correlations_in_folder(folder1_path, min_rows)

(spearman_correlations_folder2, kendall_correlations_folder2,
 all_spearman_matrices_folder2, all_kendall_matrices_folder2,
 count_spearman2, count_kendall2) = calculate_nonlinear_correlations_in_folder(folder2_path, min_rows)

first_spearman_matrix_folder1 = all_spearman_matrices_folder1[0]
first_spearman_matrix_folder2 = all_spearman_matrices_folder2[0]

first_kendall_matrix_folder1 = all_kendall_matrices_folder1[0]
first_kendall_matrix_folder2 = all_kendall_matrices_folder2[0]


# In[200]:


print("Spearman Correlations for CSV files in ADHD:")
for filename, spearman_corr in spearman_correlations_folder1.items():
    print(f"Spearman Correlation matrix for {filename}:")
    print(spearman_corr)


# In[201]:


print("Kendall Correlations for CSV files in ADHD:")
for filename, kendall_corr in kendall_correlations_folder1.items():
    print(f"Kendall Correlation matrix for {filename}:")
    print(kendall_corr)


# In[202]:


print("Spearman Correlations for CSV files in control:")
for filename, spearman_corr in spearman_correlations_folder2.items():
    print(f"Spearman Correlation matrix for {filename}:")
    print(spearman_corr)


# In[203]:


print("Kendall Correlations for CSV files in control:")
for filename, kendall_corr in kendall_correlations_folder2.items():
    print(f"Kendall Correlation matrix for {filename}:")
    print(kendall_corr)


# In[204]:


print("Number of Spearman Correlation matrices in folder 1:", count_spearman1)
print("Number of Kendall Correlation matrices in folder 1:", count_kendall1)
print("Number of Spearman Correlation matrices in folder 2:", count_spearman2)
print("Number of Kendall Correlation matrices in folder 2:", count_kendall2)


# # HEATMAP

# In[284]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


def plot_heatmap(correlation_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title(title)
    plt.show()

for filename, spearman_corr in spearman_correlations_folder1.items():
    plot_heatmap(spearman_corr, f"Spearman Correlation Heatmap ADHD - {filename}")


for filename, kendall_corr in kendall_correlations_folder1.items():
    plot_heatmap(kendall_corr, f"Kendall Correlation Heatmap ADHD - {filename}")


for filename, spearman_corr in spearman_correlations_folder2.items():
    plot_heatmap(spearman_corr, f"Spearman Correlation Heatmap CONTROL - {filename}")

    
for filename, kendall_corr in kendall_correlations_folder2.items():
    plot_heatmap(kendall_corr, f"Kendall Correlation Heatmap CONTROL- {filename}")


# # HEADER UPDATE 

# In[206]:


import os
import pandas as pd

header_mapping = {
    '0': 'Fp1',
    '1': 'Fp2',
    '2': 'F3',
    '3': 'F4',
    '4': 'C3',
    '5': 'C4',
    '6': 'P3',
    '7': 'P4',
    '8': 'O1',
    '9': 'O2',
    '10': 'F7',
    '11': 'F8',
    '12': 'T7',
    '13': 'T8',
    '14': 'P7',
    '15': 'P8',
    '16': 'Fz',
    '17': 'Cz',
    '18': 'Pz',
}


def update_csv_headers(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [header_mapping.get(col, col) for col in df.columns]
    df.to_csv(csv_path, index=False)
    

folder1_path = 'D:/Sem5/ML LAB TEST/CSV/ADHD'
folder2_path = 'D:/Sem5/ML LAB TEST/CSV/Control part'


def add_column_y(csv_path, value):
    df = pd.read_csv(csv_path)
    df['y'] = value
    df.to_csv(csv_path, index=False)


for filename in os.listdir(folder1_path):
    if filename.endswith(".csv"):
        csv_path = os.path.join(folder1_path, filename)
        update_csv_headers(csv_path)
        add_column_y(csv_path, 0) 


for filename in os.listdir(folder2_path):
    if filename.endswith(".csv"):
        csv_path = os.path.join(folder2_path, filename)
        update_csv_headers(csv_path)
        add_column_y(csv_path, 1)  


# # HEATMAP FOR A PARTICULAR SUBJECT 

# In[285]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


def plot_heatmap(correlation_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title(title)
    plt.show()


def select_folder():
    print("Select a folder:")
    print("1: ADHD")
    print("2: CONTROL PART")

    try:
        folder = int(input("Enter the number of the folder you want to use: "))
        if folder not in [1, 2]:
            print("Invalid folder number. Please enter 1 or 2.")
            return None
        return folder
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None


def select_csv_file(folder):
    folder_name = "ADHD" if folder == 1 else "Control Part"
    folder_path = "D:/Sem5/ML LAB TEST/CSV"
    csv_files = [file for file in os.listdir(os.path.join(folder_path, folder_name)) if file.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {folder_name}.")
        return None

    print(f"Select a CSV file from {folder_name}:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}: {file}")

    try:
        choice = int(input("Enter the number of the CSV file you want to plot: ")) - 1
        if 0 <= choice < len(csv_files):
            selected_filename = csv_files[choice]
            return selected_filename
        else:
            print("Invalid choice. Please enter a valid number.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def plot_correlation_heatmaps(folder, selected_filename):
    folder_name = "ADHD" if folder == 1 else "CONTROL PATH"


    spearman_correlation_matrix = spearman_correlations_folder1.get(selected_filename) if folder == 1 else spearman_correlations_folder2.get(selected_filename)
    if spearman_correlation_matrix is not None:
        spearman_title = f"Spearman Correlation Heatmap {'ADHD' if folder == 1 else 'CONTROL'} - {selected_filename}"
        plot_heatmap(spearman_correlation_matrix, spearman_title)


    kendall_correlation_matrix = kendall_correlations_folder1.get(selected_filename) if folder == 1 else kendall_correlations_folder2.get(selected_filename)
    if kendall_correlation_matrix is not None:
        kendall_title = f"Kendall Correlation Heatmap {'ADHD' if folder == 1 else 'CONTROL'} - {selected_filename}"
        plot_heatmap(kendall_correlation_matrix, kendall_title)


folder = select_folder()  
if folder is not None:
    selected_filename = select_csv_file(folder)
    if selected_filename is not None:
        plot_correlation_heatmaps(folder, selected_filename)


# # TOPOPLOT

# In[208]:


pip install mne


# In[179]:


pip install eeg_positions


# In[209]:


import numpy as np
import pandas as pd
import mne
from eeg_positions import get_elec_coords
import matplotlib.pyplot as plt
from mne import create_info
from mne.viz import plot_topomap
import numpy as np


# In[330]:


import os
import pandas as pd


def select_csv_file(folder_name):
    folder_path = "D:/Sem5/ML LAB TEST/CSV"
    csv_files = [file for file in os.listdir(os.path.join(folder_path, folder_name)) if file.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {folder_name}.")
        return None

    while True:
        print(f"Select a CSV file from {folder_name}:")

        for i, file in enumerate(csv_files):
            print(f"{i + 1}: {file}")

        try:
            choice = int(input("Enter the number of the CSV file you want to select (0 to quit): "))
            if choice == 0:
                return None
            elif 0 < choice <= len(csv_files):
                selected_filename = csv_files[choice - 1]
                return selected_filename
            else:
                print("Invalid choice. Please enter a valid number or 0 to quit.")
        except ValueError:
            print("Invalid input. Please enter a number or 0 to quit.")


folder_name = ""

while True:
    try:
        choice = int(input("Select an option:\n1. ADHD\n2. CONTROL PATH\nEnter your choice (1 or 2): "))
        if choice == 1:
            folder_name = "ADHD"
            selected_filename = select_csv_file(folder_name)  # Dynamically select from folder 'ADHD'
            break
        elif choice == 2:
            folder_name = "Control part"  # Set folder_name to empty string for file dialog
            selected_filename = select_csv_file(folder_name)  # Dynamically select from folder 'ADHD'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter 1 or 2.")

if selected_filename:
    folder_path = "D:/Sem5/ML LAB TEST/CSV"
    df = pd.read_csv(os.path.join(folder_path, folder_name, selected_filename))
    print(f"CSV file '{selected_filename}' loaded successfully!")
else:
    print("No file selected.")


# In[331]:


df.head()


# In[332]:


info = df.columns.values[:-1]
data_min = df.min(axis=0)[:-1]


# In[333]:


coords = get_elec_coords(system="1020",dim="2d").set_index("label")
coords['xy']=coords.apply(lambda x : (x.x,x.y),axis=1)
coords['xy']


# In[334]:


pos_arr = [coords['xy'][x] for x in info]
print(len(pos_arr))


# In[335]:


pos_array=np.array(pos_arr)


# In[336]:


column=['Fz','Cz','Pz','C3','T3','C4','T4','Fp1','Fp2','F3','F4','F7','F8','P3','P4','T5','T6','O1','O2']


info = create_info(ch_names=column, sfreq=1000, ch_types='eeg')
info.set_montage('standard_1020')
data = data_min.to_numpy()
fig, ax = plt.subplots(figsize=(6, 6))
plot_topomap(data, info,axes=ax,cmap='gist_rainbow_r',sensors='ko')
plt.show()


# # MEDIAN 

# In[293]:


import pandas as pd
import numpy as np
import os

directory_path = "D:/Sem5/ML LAB TEST/CSV"


median_values_adhd = []
median_values_control = []

for dir_name in os.listdir(directory_path):
    dir_path = os.path.join(directory_path, dir_name) 
    if os.path.isdir(dir_path):
        is_adhd_group = "ADHD" in dir_name

        
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    df = df.iloc[:, :19]

                    
                    if is_adhd_group:
                        median_values_adhd.append(df.median(axis=0))
                    else:
                        median_values_control.append(df.median(axis=0))

                    
                    print(f"Median values (excluding columns 19 and beyond) for {filename} in group {dir_name}:\n{df.median(axis=0)}")
                except FileNotFoundError:
                    print(f"File {filename} not found.")
                except Exception as e:
                    print(f"An error occurred for {filename} in group {dir_name}: {str(e)}")

adhd_matrix = np.array(median_values_adhd)
control_matrix = np.array(median_values_control)


# In[294]:


adhd_matrix = np.array(median_values_adhd)
control_matrix = np.array(median_values_control)


# In[295]:


print(adhd_matrix)


# In[296]:


print(control_matrix)


# In[297]:


adhd_shape = adhd_matrix.shape
control_shape = control_matrix.shape

print("Dimensions of ADHD Matrix:", adhd_shape)
print("Dimensions of Control Matrix:", control_shape)


# # SIMILARITY MATRIX AND LAPLACIAN MATRIX

# In[298]:


import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances

adhd_matrix = np.array(median_values_adhd)
control_matrix = np.array(median_values_control)

bandwidth = 1.0  


pairwise_distances_adhd = euclidean_distances(adhd_matrix)
similarity_matrix_adhd = np.exp(- (pairwise_distances_adhd ** 2) / (2 * bandwidth ** 2))

pairwise_distances_control = euclidean_distances(control_matrix)
similarity_matrix_control = np.exp(- (pairwise_distances_control ** 2) / (2 * bandwidth ** 2))


def laplacian_matrix(similarity_matrix):
    # Calculate the degree matrix
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    
    # Calculate the Laplacian matrix
    laplacian = degree_matrix - similarity_matrix
    
    return laplacian

laplacian_adhd = laplacian_matrix(similarity_matrix_adhd)
laplacian_control = laplacian_matrix(similarity_matrix_control)


# In[299]:


print(laplacian_adhd)


# # SPECTRAL CLUSTERING

# In[300]:


from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

laplacian_adhd = laplacian_matrix(similarity_matrix_adhd)
laplacian_control = laplacian_matrix(similarity_matrix_control)

num_clusters = 2

eigenvalues_adhd, eigenvectors_adhd = eigh(laplacian_adhd)
eigenvalues_control, eigenvectors_control = eigh(laplacian_control)

smallest_eigenvalues_adhd = eigenvalues_adhd[:num_clusters]
smallest_eigenvectors_adhd = eigenvectors_adhd[:, :num_clusters]

smallest_eigenvalues_control = eigenvalues_control[:num_clusters]
smallest_eigenvectors_control = eigenvectors_control[:, :num_clusters]


embedding_adhd = PCA(n_components=num_clusters).fit_transform(smallest_eigenvectors_adhd)
embedding_control = PCA(n_components=num_clusters).fit_transform(smallest_eigenvectors_control)


cluster_labels_adhd = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit_predict(embedding_adhd)
cluster_labels_control = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit_predict(embedding_control)


# In[301]:


print(len(cluster_labels_adhd))


# In[302]:


print(len(cluster_labels_control))


# In[303]:


import matplotlib.pyplot as plt

# Scatter plot for ADHD data
plt.figure(figsize=(8, 6))
plt.scatter(embedding_adhd[:, 0], embedding_adhd[:, 1], c=cluster_labels_adhd, cmap='viridis')
plt.title('Spectral Clustering Results for ADHD')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar()
plt.show()

# Scatter plot for control data
plt.figure(figsize=(8, 6))
plt.scatter(embedding_control[:, 0], embedding_control[:, 1], c=cluster_labels_control, cmap='viridis')
plt.title('Spectral Clustering Results for Control')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar()
plt.show()


# # RANK

# In[304]:


import numpy as np

# Calculate the rank of the median matrix

adhd_rank = np.linalg.matrix_rank(adhd_matrix)
control_rank = np.linalg.matrix_rank(control_matrix)

# Print the rank
print(f"The rank of the median matrix is: {adhd_rank}")
print(f"The rank of the median matrix is: {control_rank}")


# If the result of calculating the rank of your 121x19 median matrix is 19, it means that all 19 columns in your matrix are linearly independent. In other words, there are no linear dependencies or redundancies among these columns.
# 
# Having a rank equal to the number of columns (19 in this case) implies that each column provides unique and essential information

# # MODEL BUILDING

# In[305]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


# In[306]:


labels_adhd = [0] * len(adhd_matrix)
labels_control = [1] * len(control_matrix)

# Combine the labels into a single target array 'y'
y = np.concatenate((labels_adhd, labels_control))
print(y)
# Perform feature selection using SelectKBest
num_features_to_select = 10
selector = SelectKBest(f_classif, k=num_features_to_select)
median_matrix_selected = selector.fit_transform(np.concatenate((adhd_matrix, control_matrix)), y)


# In[307]:


print(median_matrix_selected)


# In[308]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(median_matrix_selected, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', C=1.0)  # You can choose different kernel functions and C values based on your data and problem.

svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# In[ ]:





# In[ ]:




