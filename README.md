**EEG data analysis and modelling**
<pre>
--Downloaded the adhd dataset from IEEE data portal<br />
    The dataset includes EEG data of 121 subjects of which 60 Adhd and 61 are control<br />
    Preprocessing:<br />
        All files are mat files, first converted into CSV.<br />
        Normalized the no of rows.<br />
    Found the nonlinear correlation coefficient for every column with every other column. As a result, got a 19*19 matrix for each subject.<br />
    Analysed the values.<br />
    Plot topoplot for one subject from adhd and one from control and find the inference.<br />
    For each class,<br />
      Found median value for each cell, As a result one matrix for each class is obtained.<br />
    Applied spectral clustering on these matrices and cluster the electrodes.<br />
    Applied spectral clustering based on lobes and hemispheres.<br />
    Finding the rank of matrices and infer the values.<br />
</pre>

Later, the analysis and modeling were translated to Streamlit for creating a User Interface representation.
ADHD , Control Folder contains the csv files.
EEG_Analysis_with_interface.py is the code which also contains the streamlit part.
EEG_Analysis.py contains only the analysis and machine learning part without UI.

