# Protein Dynamics-Based Deep Learning Model
(1)	Data – protein sequence, protein function. 

(1.1)	AA index – values for AAs. 19.npy – perhaps a wrapper encoding the 19 properties

(1.2)	Actual protein system folders. Here we can, if we want, include a seq.csv master csv file that contains variant, variant number, “score” and the AA for each position. Must also have wt.pdb

(1.3)	The most important folder in this sub folder is “Processed_data”. It contains test.csv and train.csv, made up of our pre-separated training and testing sets from the main .csv of (1.2). It also contains the K-fold sets of “fold_X_train.csv” and “fold_X_test.csv”. You need to curate this manually.

(1.4)	DCI folder – contains a folder for every single position where the folder number is the perturbed position. Each of these numbered folders contains a single .csv file with the DCI responses, the numbers must match the seq.csv file 

** change the environment path correctly in the environment.yml file **

For starters, use CNN. Use the submit.sh file. For the “protein” flag it must match the folder name of the protein under the “Data” folder. Here make sure you change the .py called to the architecture you want to use. 

In the “endpoints” folder will be the .py codes for many other model architectures to train with corresponding submit_X.sh files.

If I want to try over many params, use the “sweep” version. 

For initial runs, “results” and “analyses” folders should populate on their own. 

This will train the model on the supplied data. 
