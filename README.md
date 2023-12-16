**EEG data analysis and modelling**

--Downloaded the adhd dataset from IEEE data portal
    The dataset includes EEG data of 121 subjects of which 60 Adhd and 61 are control
    Preprocessing:
        All files are mat files, first converted into CSV.
        Normalized the no of rows.
    Found the nonlinear correlation coefficient for every column with every other column. As a result, got a 19*19 matrix for each subject.
    Analysed the values.
    Plot topoplot for one subject from adhd and one from control and find the inference.
    For each class,
      Found median value for each cell, As a result one matrix for each class is obtained.
    Applied spectral clustering on these matrices and cluster the electrodes.
    Applied spectral clustering based on lobes and hemispheres.
    Finding the rank of matrices and infer the values.
