#!/usr/bin/python

### Import used libraries
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn import svm, datasets, preprocessing, metrics
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import f1_score, make_scorer,classification_report
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import pprint
import datetime
import winsound     
from time import time   

## Header
print "\nPROJECT REPORT - MODULE 5 MACHINE LEARNING"
print "\t by Jonas Dinesen, January 2017"

#########################################################################
### General 
#########################################################################
print "\nA.INITIALIZATION"
print "\tExplore, Select features and remove outliers"
t0 = time()

## Setup PrettyPrint  
pp = pprint.PrettyPrinter(indent=4, width=70, depth=4)

## Function for drawing plots - display POI as stars
def drawPlot(plotvalues,poi,f1_name,f2_name,title,log=False):
	for point in plotvalues:
		poi_state = point[0]
		f1_value = point[1]
		f2_value = point[2]
		color = point[3]
		# Draw Log scale plot - Add 1 to values which are 0 to still be displayed in log plot
		if log:
			if f1_value == 0:
				f1_value = 1
		 	if f2_value == 0:
		 		f2_value = 1
			if poi:
				if poi_state == 1:
					plt.scatter( np.log(f1_value), np.log(f2_value), color="y", marker="*" )
				else:
					plt.scatter( np.log(f1_value), np.log(f2_value), color=color  )
			else:
				plt.scatter( np.log(f1_value), np.log(f2_value), color=color  )
		# Draw normal plot
		else: 
			if poi:
				if poi_state == 1:
					plt.scatter( f1_value, f2_value, color="y", marker="*" )
				else:
					plt.scatter( f1_value, f2_value, color=color  )
			else:
				plt.scatter( f1_value, f2_value, color=color  )
	plt.xlabel(f1_name)
	plt.ylabel(f2_name)
	plt.title(title)
	plt.show()

## Create simple plot
def simplePlot(f1_name,f1_index,f2_name,f2_index,log=False):
	plotOutliers = []
	for point in data:
		f1_value = point[f1_index]
		f2_value = point[f2_index]
		plotOutliers.append([point[0],f1_value,f2_value,"b"])
	drawPlot(plotOutliers,True,f1_name,f2_name,title="QuickPlot",log=log)

#########################################################################
### Task 1: Select features 

## Combine all features in one list
poi = ['poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # 'email_adresses' has been removed due to no text search will take place
##  Combine above lists
features_list = poi + financial_features + email_features

### Load the dictionary containing the dataset and extract features and labels
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict #dict
data = featureFormat(my_dataset, features_list, sort_keys = True) #numpy array
labels, features = targetFeatureSplit(data) #lists

## Count number of POIs:
def countPOIs():
	pois_no = 0
	for entry in data:
		if entry[0] == 1.0:
			pois_no += 1
	return pois_no
pois = countPOIs()

print "\nStart data"
print "\tNumber of People in dataset:",len(data_dict)
print "\tNumber of POIs in dataset:", pois
print "\tNumber of features to begin with:",len(features_list)

### Below checking for features with many missing/zero values 
## the concern is that features with too many NaN/zero values will skew the data
## The rate is displayed in percentage 
def check_nans(data):
	nan_list = {}
	for feat in features_list:
		nan_list[feat] = [0.0,0.0,0.0] 
	count_feats = len(features_list)
	count_pois = (data[:,0] == float(1.0)).sum()
	i = 0
	for i in range(0,count_feats):
		feat = features_list[i]
		counter = 0
		for line in data:
			counter +=1
			if line[i] == 0.0:
				nan_list[feat][0] += 1
				if line[0] == 1.0:
					nan_list[feat][1] +=1
				else:
					nan_list[feat][2] +=1
		nan_list[feat][0] = round(1- (float(nan_list[feat][0]) / float(counter)),2)
		nan_list[feat][1] = round(1- (float(nan_list[feat][1]) / float(count_pois)),2)
		nan_list[feat][2] = round(1- (float(nan_list[feat][2]) / float(counter-count_pois)),2)
	print "\nChecking NaNs:"
	print "\tPOIs:",count_pois, "All:",counter,"Features:",count_feats
	print "\tFeature non-NaNs: ALL, POIs, Non_POIs"
	pp.pprint (nan_list)
## Run the function
check_nans(data) # FINAL

## Based on this list - checking the amount of 0/NaN values and the percentage of POIs having the value I decide to remove the following features for first run:
## with the criteria of too few occurences for POI or overall "bonus" (director_fees, loan_advances(?), restricted_stock_deferred)
remove = ['director_fees', 'loan_advances', 'restricted_stock_deferred']
for feat in remove:
	features_list.remove(feat)

### Extract features and labels from dataset for local testing (again)
data = featureFormat(my_dataset, features_list, sort_keys = True) #numpy array
labels, features = targetFeatureSplit(data)

#########################################################################
### Task 2: Remove outliers
#########################################################################

## Find (possible) outliers for specific features and create plot with (possible) outliers marked
def find_outliers(f1_name,f1_index,f2_name,f2_index,f1_thresh,f2_thresh):
	outliers = []
	plotOutliers = []
	
	## Check features for exceeding threshold
	for point in data:
		f1_value = point[f1_index]
		f2_value = point[f2_index]
		## Plot above threshold values in red and add to list of outliers
		if point[f1_index] > f1_thresh and point[f2_index] > f2_thresh:
			plotOutliers.append([point[0],f1_value,f2_value,"r"])
			## Print name and amount of outlier
			for note in data_dict:
				if data_dict[note][f1_name] == point[f1_index]:
					value = f1_name, data_dict[note][f1_name]#, f2_name,data_dict[note][f2_name]
					outliers.append(["High Amoun",note,value,point[0]])
		elif point[f1_index] == 0:
			plotOutliers.append([point[0],f1_value,f2_value,"y"])
		else:
			plotOutliers.append([point[0],f1_value,f2_value,"b"])

	## Check all features with value NaN or with only one word in name
	for entry in data_dict:
		if data_dict[entry][f1_name] == "NaN" and data_dict[entry][f2_name] == "NaN":
			outliers.append(["Value NaNs",entry])
		if (' ' in entry) ==False:
			outliers.append(["OneWord Na",entry])

	## Plot to see outliers 
	drawPlot(plotOutliers,True,f1_name,f2_name,title="Spot Outliers")

	## Return list of outliers
	return outliers

### Check for outliers
pp.pprint(find_outliers("salary",1,"expenses",7,2000000,0))

# Outlier - manually selected
selected_outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']

def missing_salary():
	no_salary_lst = []
	for entry in data_dict:
		if data_dict[entry]['salary'] == 'NaN':
			no_salary_lst.append(entry)
	return no_salary_lst

no_salary_lst = missing_salary()

# Function to Remove outliers deemed necessary to remove through above
def remove_outliers(outliers):
	removedOutliers = []
	print "\nStatus for removal:" #Final
	if len(outliers) > 0:
		for point in outliers:
			try:
				data_dict.pop( point, 0 )
				print "\tRemoved:",point 
			except: 
				print "\tFailed to remove:", point
	else: 
		print "\tNo outliers to remove"
	return data_dict

# Send selected outliers for being removed 
data_dict = remove_outliers(selected_outliers)
data_dict = remove_outliers(no_salary_lst)

# Recreate features and labels
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

## Plotting several charts 
#print "\n",features_list[1], features_list[4]
simplePlot("salary",1,"bonus",4,True) #FINAL / 8 exercies_stock_options + 9 Other..10=
#simplePlot("salary",1,"defferal_payments",2)
#simplePlot("salary",1,"total_stock_value",6)
#simplePlot("salary",1,"from_this_person_to_poi",15)

#########################################################################
### Task 3: Create new feature(s)
#########################################################################

# Creating fractions of emails send/received 
def computeFraction( messages, all_messages ):
	# Divide nominator with denominator - if nominator is NaN set to zero
	if messages != "NaN":
		fraction = float(messages)/float(all_messages)
	else:
		fraction = 0
	return fraction

submit_dict = {}
for name in data_dict:
	data_point = data_dict[name]
	# New datapoint - Fractions for From POI messages
	from_messages = data_point["from_messages"]
	to_messages = data_point["to_messages"]
	fraction_to_vs_from = computeFraction( to_messages, from_messages )
	# Inserting value
	data_point["fraction_to_vs_from"] = fraction_to_vs_from
	# New datapoint - Fractions for From POI messages
	from_poi_to_this_person = data_point["from_poi_to_this_person"]
	to_messages = data_point["to_messages"]
	fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
	# Inserting value
	data_point["fraction_from_poi"] = fraction_from_poi
	# New datapoint - Fraction from person to POI
	from_this_person_to_poi = data_point["from_this_person_to_poi"]
	from_messages = data_point["from_messages"]
	fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
	submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,"from_this_person_to_poi":fraction_to_poi}
	data_point["fraction_to_poi"] = fraction_to_poi

def submitDict():
    return submit_dict

## Create new data list with new features
new_features = ['fraction_to_poi','fraction_from_poi','fraction_to_vs_from'] 
features_list = features_list + new_features

# Recreate features and labels
labels, features = targetFeatureSplit(data)
data = featureFormat(data_dict, features_list, sort_keys = True)
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

## Function for calculating correlation (Pearsons R) between each feature
def correlation():
	corr = np.corrcoef(data,rowvar=0)
	np.set_printoptions(precision=2,threshold=np.inf)
	print "\nCorrelation between all features:" 
	for i in range(len(corr)):
		output = features_list[i], corr[i]
		print "\t", output[0],":",output[1]
correlation() #FINAL

## The correlation chart gives an overview of how the features are interrelated. Three relations high: "total_payments & other" (0.83), 
## "to_messages & shared_receipt_with_poi" (0.87), and "total_stock_value & restricted_stock" (0.97) - I decide to remove: other, to_messages, and restricted_stock 
#remove_features = ['other', 'to_messages', 'restricted_stock'] 
#features_list = [x for x in features_list if x not in ['other', 'to_messages', 'restricted_stock']]

# Recreate features and labels
labels, features = targetFeatureSplit(data)
data = featureFormat(data_dict, features_list, sort_keys = False)
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

## Check again number of features and POIs and how the data distribute
check_nans(data) #FINAL

### See new features in plot Plot
# Plotter made to receive two indexes and to remove 0 / NaN values
def plotter(f1_index,f2_index,f1_name,f2_name,title):
	plotData= []
	for point in data:
		f1_value = point[f1_index]
		f2_value = point[f2_index]
		if not(f1_value < 1 or f2_value == 0):
			plotData.append([point[0],f1_value,f2_value,"b"])
	drawPlot(plotData,True,f1_name,f2_name,title=title)

## Plot new features
#print "\n",features_list[1], features_list[14]
#plotter(1,14,"salary","to_poi","Fraction to POI")
#plotter(2,16,"salary","from_poi","Fraction from POI")
#plotter(2,17,"salary","to_vs_from","Fraction To vs From")

pois = countPOIs()
## List all features
print "\nUpdated list of features and POIs"
print "\tNumber of persons:",len(data_dict)
print "\tNumber of POIs:",pois
print "\tNumber of features:",len(features_list)
print "\nList of features:"
print "\t", features_list 

#########################################################################
### Task 4: Try a varity of classifiers
#########################################################################

clf = ""
###### Naive-Bayes / Gaussian #########################
def nb_clf(features_train, labels_train):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	# Create Naive-Bayes classifier
	clf = clf.fit(features_train, labels_train) # set features and labels
	return clf

############## SVM ######################
def svm_clf(features_train, labels_train):
	from sklearn.svm import SVC
	clf = SVC(kernel="rbf")
	# Create SVM classifier
	clf = clf.fit(features_train, labels_train) 
	return clf

############## Decision Tree ######################
def dt_clf(features_train, labels_train):
	from sklearn import tree
	clf = tree.DecisionTreeClassifier(min_samples_split=10,max_depth=4)
	# Create Decision tree classifier
	clf = clf.fit(features_train, labels_train)
	return clf

############## Default ##################
### Accuracy Score
from sklearn.metrics import accuracy_score
def accuracy(pred, labels_test):
	acc = accuracy_score(pred, labels_test)
	return acc

## RUNtime ###
def runtime(type):
	name = None
	if type == "dt":
		clf = dt_clf(features_train, labels_train)
		name = "DecisionTree"
	if type == "nb": 
		clf = nb_clf(features_train, labels_train)
		name = "GaussianNB"
	if type == "svm":
		clf = svm_clf(features_train, labels_train)
		name = "Support Vector Machine"

	# Get prediction
	pred = clf.predict(features_test)
	# Print accuracy 
	acc = accuracy(pred,labels_test)
	print "\nAccuracy for",name
	print "\t",acc

## Run check of different classifiers
runtime("dt")
runtime("nb")
runtime("svm")

train_time = time()- t0
print "\nExploration part done. Time:", train_time

#########################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 
#########################################################################

print "\nB. GRIDSEARCHCV"
print "\tFinding the optimal Clf"
t0 = time()

def gridSearcher():
	## Set scaler
	scaler = preprocessing.StandardScaler() #preprocessing.MinMaxScaler() 

	## Make scorer
	#score  = make_scorer(custom_scorer, greater_is_better=True)

	## Select K best features
	k_filter = SelectKBest()

	## Prepare stratified shuffle split
	sss = StratifiedShuffleSplit(1000, test_size=0.2,random_state=42) # Final set to 1000
	sss.get_n_splits(features_train, labels_train) #(features,labels) 

	## Setup Classifiers
	clf_svc = svm.SVC()
	clf_dt = tree.DecisionTreeClassifier()
	clf_nb = GaussianNB()
	clf_knn = KNeighborsClassifier()
	clf_ab = AdaBoostClassifier()
	clf_rf = RandomForestClassifier()

	## Setup PCA filter
	pca_filter = PCA()

	## Set parameters to go through
	pl_params = {'SelectKBest__k': [7],#[7,8,9,10]
	              'pca__n_components': [2], #[1,2,3]
#	              'knn__n_neighbors': [1,2,3,4],'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'], 'knn__leaf_size':[3,4,5],'knn__weights':['uniform','distance']
#	              'dt__min_samples_leaf': [8, 16, 32],'dt__criterion':['gini','entropy'],'dt__random_state':[42]
#	              'ab__n_estimators': [30,50,100],'ab__learning_rate':[1,2,3]				
#	              'rf__n_estimators': [5,10,15], 'rf__min_samples_leaf':[10,50,100]#,'rf__criterion': ['gini','entropy'],'max_features':['auto','sqrt','log2',None]
	              'svc__kernel':['rbf'], 'svc__C':[9],'svc__gamma': [0.018], 'svc__class_weight':['balanced'],'svc__max_iter':[-1],'svc__tol':[1] #0.9,1,1.2 
	             } 
	## Create pipeline - after each clf is noted how well they fared in test run
	pipeline = Pipeline(steps=[
#		('MinMaxScaler', scaler), 
		('StandardScaler',scaler),	
		('SelectKBest', k_filter),
		('pca', pca_filter),
#		('nb', clf_nb) #Precision: 0.35041	Recall: 0.19150	
#		('dt', clf_dt) #Precision: 0.32196	Recall: 0.22650
#		('knn', clf_knn) #Precision: 0.23293	Recall: 0.22350
#		('ab', clf_ab) #Precision: 0.27549	Recall: 0.23100
#		('rf', clf_rf) #Precision: 0.15789	Recall: 0.00450
		('svc', clf_svc) #Precision: 0.34345	Recall: 0.59400
		])
	## Run GridSearchSV
	gs = GridSearchCV(pipeline, pl_params, scoring='f1', cv=sss) 
	gs.fit(features_train,labels_train)

	## The optimal model selected by GridSearchCV
	best_clf = gs.best_estimator_

	## Calculate prediction
	prediction = gs.predict(features_test)
	
	## Print output of GridSearchCV
	print '\nScores'
	print '\tPrecision:', metrics.precision_score(labels_test, prediction)
	print '\tRecall:', metrics.recall_score(labels_test, prediction)
	print '\tF1 Score:', metrics.f1_score(labels_test, prediction)
	print '\nBest estimator'
	print '\t', best_clf
	print '\nBest score: %0.3f' % gs.best_score_
	print '\nBest parameters set:'
	best_parameters = gs.best_estimator_.get_params()
	for param_name in sorted(pl_params.keys()):
		print '\t%s: %r' % (param_name, best_parameters[param_name])
	print '\n', "Classification Report:"
	print classification_report(labels_test, prediction)

	features_selected_bool = gs.best_estimator_.named_steps['SelectKBest'].get_support()
	features_selected_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
	print "\t",features_selected_list

	return best_clf

## Run GridsearchCV
clf = gridSearcher() # FINAL

train_time = time()- t0
print "\tGridSearchCV finished. Time:", train_time


#########################################################################
### Task 6: Dump classifier, dataset, and features_list 

t0 = time()
print "\nC.TEST and EXPORT"
## Run test 
test_classifier(clf, my_dataset, features_list, folds = 1000)
## Dump data
dump_classifier_and_data(clf, my_dataset, features_list) #FINAL 

## Timing
train_time = time()- t0
print "\tOptimal Classifier calculated. Time:", train_time

## Notify with beep when script has finished running
winsound.Beep(440, 650) # frequency, duration