#Author: Revel Etheridge
#Date: 01-28-2025
#Title: locationCrossing.R
#Purpose: Read in csv's, label and save new data set, call python file to map new data set

#Importing Libraries
library(reticulate)

#Accessing python
use_python(use_python("/usr/local/bin/python3"))

#Reading in sample data
sample = read.csv('reticSample.csv')

#Saving data to be mapped to csv
write.csv(sample, "fresh.csv")

#Calling python file to begin mapping process
py_run_file('pyMap.py')

#Returning after mapping is complete
print("done")