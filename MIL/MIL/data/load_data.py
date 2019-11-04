import pickle
from scipy.io import loadmat
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import ipaddress

def load_data(data):
    
    if data == 'musk1_scaled':  
        filename_bag = 'D:/Projects/MILpy/data/musk1_scaled/Bag2_mus_escal.mat'
        filename_labels = 'D:/Projects/MILpy/data/musk1_scaled/bagI_mus_escal.mat'
        X_g = loadmat('D:/Projects/MILpy/data/musk1_scaled/X_mus_escal.mat')
    elif data == 'data_gauss': 
        filename_bag = 'D:/Projects/MILpy/data/gauss_data/bag_g.mat'
        filename_labels = 'D:/Projects/MILpy/data/gauss_data/bagI_gauss.mat'
        X_g = loadmat('D:/Projects/MILpy/data/gauss_data/X_g.mat')
    elif data == 'musk1_original':
        filename_bag = 'D:/Projects/MILpy/data/musk1_unscaled/Bag2_musk_original.mat'
        filename_labels = 'D:/Projects/MILpy/data/musk1_unscaled/bagI_musk1_original.mat'
        X_g = loadmat('D:/Projects/MILpy/data/musk1_unscaled/X_musk_original.mat')
    elif data== 'robot':
        bags,labels = loaddata()
        return bags,labels
    elif data == 'robot_test':
        bags,labels = loaddata_test()
        return bags,labels
    else:
        file = data
        filename_bag = 'D:/Projects/MILpy/data/'+file+'/Bag2.mat'
        filename_labels = 'D:/Projects/MILpy/data/'+file+'/bagI.mat'
        X_g = loadmat('D:/Projects/MILpy/data/'+file+'/X.mat')
    
        
    bag_g = loadmat(filename_bag)
    labels = loadmat(filename_labels)
    try: 
        Bag = bag_g['Bag2']
    except KeyError: 
        Bag = bag_g['Bag']

    labels = labels['bagI']
    X = X_g['X']
    Bag = np.squeeze(Bag-1)
    nrobags = max(Bag+1)
    bags = []
    for i in range(0,nrobags): 
        index = np.where( Bag == i )
        bag = X[index]
        bags.append(bag)

    return bags,labels

def loaddata():
    bags =[]
    labels = []
    filename = 'D:/NewTrain/csv/NewTrain_0.csv'
    df = read_csv(filename)
    grouped= df.groupby(['AccountID'])
    for name,group in grouped:
         bag = []
         label = np.array(group.pop('Target'))
         #group['IP'] = group['IP'].replace('unknown', 0)
         #group['IP'] = group['IP'].map(lambda x: int(ipaddress.IPv4Address(x)) if x !='unknown' else 0)
         group = group.drop(columns=['IP'])
         group = group.drop(columns=['AccountID'])
         group = group.drop(columns=['TransID'])
         group = group.drop(columns=['Year'])
         group =  group.drop(columns=['Month'])
         group =  group.drop(columns=['Day'])
         group =  group.drop(columns=['Hour'])
         group =  group.drop(columns=['Minute'])
         group =  group.drop(columns=['Second'])
         label_encoder = LabelEncoder() 
         group['Browser']= label_encoder.fit_transform(group['Browser']) 
         group = StandardScaler().fit_transform(group)
         for obj in group:
             bag.append(obj)
         bags.append(bag) 
         labels.append([label[0]])

    filename = 'D:/NewTrain/csv/NewTrain_1.csv'
    df = read_csv(filename)
    grouped= df.groupby(['AccountID'])
    for name,group in grouped:
         bag = []
         label = np.array(group.pop('Target'))
         #group['IP'] = group['IP'].replace('unknown', 0)
         #group['IP'] = group['IP'].map(lambda x: int(ipaddress.IPv4Address(x)) if x !='unknown' else 0)
         group = group.drop(columns=['IP'])
         group = group.drop(columns=['AccountID'])
         group = group.drop(columns=['TransID'])
         group = group.drop(columns=['Year'])
         group =  group.drop(columns=['Month'])
         group =  group.drop(columns=['Day'])
         group =  group.drop(columns=['Hour'])
         group =  group.drop(columns=['Minute'])
         group =  group.drop(columns=['Second'])
         label_encoder = LabelEncoder() 
         group['Browser']= label_encoder.fit_transform(group['Browser']) 
         group = StandardScaler().fit_transform(group)
         for obj in group:
             bag.append(obj)
         bags.append(bag) 
         labels.append([label[0]])

    filename = 'D:/csv/data.dat'
    pickle.dump(bags, open(filename, 'wb'))
    filename = 'D:/csv/data_label.txt'
    pickle.dump(np.array(labels), open(filename, 'wb'))
    return bags,np.array(labels)

def loaddata_test():
    bags =[]
    labels = []
    filename = 'D:/NewTrain/verify.csv'
    df = read_csv(filename)
    grouped= df.groupby(['AccountID'])
    for name,group in grouped:
         bag = []
         label = np.array(group.pop('Target'))
         #group['IP'] = df['IP'].replace('unknown', 0)
         #group['IP'] = df['IP'].map(lambda x: int(ipaddress.IPv4Address(x)) if x !='unknown' else 0)
         group = group.drop(columns=['IP'])
         group = group.drop(columns=['AccountID'])
         group = group.drop(columns=['TransID'])
         group = group.drop(columns=['Year'])
         group =  group.drop(columns=['Month'])
         group =  group.drop(columns=['Day'])
         group =  group.drop(columns=['Hour'])
         group =  group.drop(columns=['Minute'])
         group =  group.drop(columns=['Second'])
         label_encoder = LabelEncoder() 
         group['Browser']= label_encoder.fit_transform(group['Browser']) 
         group = StandardScaler().fit_transform(group)
         for obj in group:
             bag.append(obj)
         bags.append(bag) 
         labels.append([label[0]])
    return bags,np.array(labels)
           


    
    
    

