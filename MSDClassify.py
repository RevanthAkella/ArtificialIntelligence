import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import Tkinter as tk
from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import shutil
import argparse
from Tkinter import *
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog

localpath = '/Users/revanth/Desktop/MSDtoCSV'


def select():
	global path
	path = tkFileDialog.askopenfilename()



def classify():
	Rock = pd.read_csv('RockDataset.csv')
	Rock = Rock.filter(['SongNumber','KeySignature','Tempo','TimeSignature','Loudness'], axis=1)
	HipHop = pd.read_csv('HipHopDataset.csv')
	HipHop = HipHop.filter(['SongNumber','KeySignature','Tempo','TimeSignature','Loudness'], axis=1)
	Dance = pd.read_csv('DanceandElectronic.csv')
	Dance = Dance.filter(['SongNumber','KeySignature','Tempo','TimeSignature','Loudness'], axis=1)
	#Data = pd.read_csv("Test.csv")
	Data = pd.read_csv(path)
	X = pd.DataFrame(Data.drop(['SongID','Hotness','Genre','ArtistName','Title'],axis=1))
	Y = Data.Genre
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
	result = []


	print "==============GaussianNB=============="
	print"\n"
	gnb = GaussianNB()

	print "================HipHop================"
	print "\n\n"

	y_pred = gnb.fit(X_train, Y_train).predict(HipHop)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, GaussianNB ="+str(accuracy))
	result.append(file)
	print"\n\n"

	print "=================Rock================="
	print"\n\n"
	y_pred = gnb.fit(X_train, Y_train).predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, GaussianNB ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"
	y_pred = gnb.fit(X_train, Y_train).predict(Dance)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, GaussianNB ="+str(accuracy))
	result.append(file)
	print"\n\n"

	print "=============SGDClassifier============"
	print"\n"
	clf = SGDClassifier(loss="hinge", penalty="l2")
	clf.fit(X_train, Y_train)

	print "=================Rock================="
	print"\n\n"
	y_pred = clf.predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, SGDClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"
	y_pred = clf.predict(HipHop)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, SGDClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"
	y_pred = clf.predict(Dance)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, SGDClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "=============KNNClassifier============"
	print"\n"
	print "=================Rock================="
	print"\n\n"
	neigh = KNeighborsClassifier(n_neighbors=4)
	y_pred = neigh.fit(X_train, Y_train).predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, KNNClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"
	y_pred = neigh.fit(X_train, Y_train).predict(HipHop)
	print y_pred 
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, KNNClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"
	y_pred = neigh.fit(X_train, Y_train).predict(Dance)
	print y_pred 
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, KNNClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "========DecisionTreeClassifier========"
	print"\n"
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, Y_train)

	print "=================Rock================="
	print"\n\n"
	y_pred = clf.predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, DecisionTreeClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"
	y_pred = clf.predict(HipHop)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, DecisionTreeClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"
	y_pred = clf.predict(Dance)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, DecisionTreeClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"


	print "========RandomForestClassifier========"
	print"\n"
	clf = RandomForestClassifier(n_estimators=4)
	clf = clf.fit(X_train, Y_train)

	print "=================Rock================="
	print"\n\n"
	y_pred = clf.predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, RandomForestClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"
	y_pred = clf.predict(HipHop)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, RandomForestClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"
	y_pred = clf.predict(Dance)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, RandomForestClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"


	print "=========ExtraTreesClassifier========="
	print"\n"

	clf = ExtraTreesClassifier(n_estimators=4)
	clf = clf.fit(X_train, Y_train)

	print "=================Rock================="
	print"\n\n"

	y_pred = clf.predict(Rock)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, ExtraTreesClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"

	y_pred = clf.predict(HipHop)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, ExtraTreesClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"

	y_pred = clf.predict(Dance)
	print y_pred
	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, ExtraTreesClassifier ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "=======NeuralNetworkClassifier========"
	print"\n"

	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 7), random_state=1)
	clf.fit(X_train, Y_train)

	print "=================Rock================="
	print"\n\n"

	y_pred = clf.predict(Rock)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, Neural Net ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"

	y_pred = clf.predict(HipHop)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, Neural Net ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print "\n\n"

	y_pred = clf.predict(Dance)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, Neural Net ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "=====SupportVectorMachineClassifier===="
	print"\n"

	clf = svm.SVC()
	clf.fit(X_train, Y_train)  

	print "=================Rock================="
	print"\n\n"

	y_pred = clf.predict(Rock)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'Rock'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Rock, SVM ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================HipHop================"
	print"\n\n"

	y_pred = clf.predict(HipHop)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'HipHop'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, HipHop, SVM ="+str(accuracy))
	result.append(file)
	print "\n\n"

	print "================Dance================="
	print"\n\n"

	y_pred = clf.predict(Dance)
	print y_pred

	tot = float(np.count_nonzero(y_pred == 'Dance'))
	pred_len = float(len(y_pred))
	accuracy = tot/pred_len
	print "Accuracy =",accuracy
	file = str("Accuracy, Dance, SVM ="+str(accuracy))
	result.append(file)
	print "\n\n"
	result = pd.DataFrame(result)
	result.to_csv('Result.csv')


def close_window():
    # start: close_window
    print("Exiting...");
    root.destroy()


root = tk.Tk()
selectedImagePanel = None

# close 'displayImagePanel' window
btnexit = Button(root, text="Exit", command=close_window)
btnexit.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")


btnselect = Button(root, text="Select a Dataset", command=select)
btnselect.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# Template Matching algo
btnclassify = Button(root, text="Classify", command=classify)
btnclassify.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()