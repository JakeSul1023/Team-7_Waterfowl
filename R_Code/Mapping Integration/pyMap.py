#Author: Revel Etheridge
#Date: 01-28-2025
#Title: pyMap.py
#Purpose: Receive longitude and latitude values from R script function call and create HTML map

#Library definitions
from __future__ import print_function
from distutils import log
from setuptools import setup, find_packages
import os
import pandas as pd

#Reading in newly saved data from csv
df = pd.read_csv('fresh.csv')

#TO DO: Incorporate basic Kepler application from Wix

print(df)

#Running HTML file with sample data
