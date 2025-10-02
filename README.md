# A Protein Dynamics-Based Deep Learning Model
(1)	Data: protein sequence and function, DCI, and pdb file. 

(1.1)	AA index: values for AAs. 19.npy 

(1.2)	“Processed_data” contains test.csv and train.csv. It also contains the K-fold sets of “train_fold#.csv” and “valid_fold#.csv”. 

(1.3)	DCI folder – contains a folder for every single position where the folder number is the perturbed position. Each of these numbered folders contains a single .csv file with the DCI responses, the numbers must match the seq.csv file 


Use the submit.sh file. For the “protein” flag it must match the folder name of the protein under the “Data” folder. Here make sure you change the .py called to the architecture you want to use. 

In the “endpoints” folder will be the .py codes for many other model architectures to train with corresponding submit_X.sh files.


For initial runs, “results” and “analysis” folders should populate on their own.  
