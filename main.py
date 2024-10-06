import csv
import numpy as np
import random
import matplotlib.pyplot as plt


data_list = []
with open("Lab1_traindata.csv", 'r') as file:
    rows = csv.reader(file)
    for row in rows:
        x1, x2, x3, y = row
        data_list.append([x1, x2, x3, y])