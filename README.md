# MSD-Genre-Classification

This is a project to classify the the tracks in the MillionSongSubset of the MillionSongDataset. The project uses features extracted from the MillionSongSubset and tests the dataset against various classifiers. The test dataset is a cleaned and pre-processed dataset for this application. The data extraction and cleaning can be done using the HDF5toCSV.py file and the HDF5 getters file. The Genres here are Dance, Rock and HipHop. The project uses Supervised Learning on a 70-30 train test split. 

The required tools for this project are:

Scikit-Learn: http://scikit-learn.org
Pandas: http://pandas.pydata.org
Tkinter:  https://docs.python.org/2/library/tkinter.html

The dataset that is used can be found on the MillionSongDataset website:

The MillionSongDataset: https://labrosa.ee.columbia.edu/millionsong/
The MillionSongSubset: https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset#subset

The MillionSong subset is a 10,000 song subset of the 1,000,000 song dataset. Its a great 1% sample for testing and tinkering before scaling up to the entire million song datset.


