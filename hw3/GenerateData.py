import numpy as np
import random
import csv

def generate_data(N):
    rand_label = []
    # here we have C=4
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.25:
            rand_label.append(1)
        elif rd <= 0.5:
            rand_label.append(2)
        elif rd <= 0.75:
            rand_label.append(3)
        else:
            rand_label.append(4)
    labels = np.array(rand_label)
    
    mean1 = (-4,3,-2)
    mean2 = (2,4,3)
    mean3 = (4,-2,-2)
    mean4 = (0,-4,-3)

    cov1 = [[10,2,-2], [2,4,0], [-2,0,10]]
    cov2 = [[4,1,0], [1,10,0], [0,0,4]]
    cov3 = [[10,-1,0], [-1,4,0], [0,0,10]]
    cov4 = [[10,0,0], [0,10,4], [0,4,10]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(mean1, cov1)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 2:
            sample = np.random.multivariate_normal(mean2, cov2)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 3:
            sample = np.random.multivariate_normal(mean3, cov3)
            rand_sample.append(sample.tolist())
        else:
            sample = np.random.multivariate_normal(mean4, cov4)
            rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)

    with open(str(N)+"_labels.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open(str(N)+"_samples.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())
    
    return samples, labels

def generate_data_2(N):
    rand_label = []
    # here we have C=4
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.22:
            rand_label.append(1) # 0.22
        elif rd <= 0.5:
            rand_label.append(2) # 0.28
        elif rd <= 0.74:
            rand_label.append(3) # 0.24
        else:
            rand_label.append(4) # 0.26
    labels = np.array(rand_label)

    mean1 = (-7,7)
    mean2 = (7,-7)
    mean3 = (-7,-7)
    mean4 = (7,7)

    cov1 = [[8,-1],[-1,8]]
    cov2 = [[7,0],[0,7]]
    cov3 = [[6,0],[0,6]]
    cov4 = [[9,2],[2,9]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(mean1, cov1)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 2:
            sample = np.random.multivariate_normal(mean2, cov2)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 3:
            sample = np.random.multivariate_normal(mean3, cov3)
            rand_sample.append(sample.tolist())
        else:
            sample = np.random.multivariate_normal(mean4, cov4)
            rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)

    with open(str(N)+"_labels_2.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open(str(N)+"_samples_2.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())
    
    return samples, labels


