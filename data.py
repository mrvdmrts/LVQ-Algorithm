from PIL import Image
import numpy as np
import os
import csv
#%% train
resim = []
resimler = []
rakamlar = []
resim_isimleri_train= []
resim_isimleri_test= []
rakamlar_train = os.listdir(str(os.getcwd()) + "/mnist-train")
rakamlar_test = os.listdir(str(os.getcwd()) + "/mnist-test")
rakamlar_train.sort()
rakamlar_test.sort()


#%% train.csv oluşturma
with open ("train.csv",mode='w') as train:
    fieldnames1 = ['category', 'pixels']
    train_writer = csv.DictWriter(train,fieldnames=fieldnames1)
    train_writer.writeheader()
    for r in rakamlar_train:
        resim_isimleri_train= os.listdir(str(os.getcwd()) + "/mnist-train/" + r)
        for i in resim_isimleri_train:
            a = Image.open(str(os.getcwd()) + "/mnist-train/" + str(r) +"/"+ str(i))
            resim = np.array(a)
            resim = resim.flatten()
            train_writer.writerow({'category': str(r),'pixels': str(resim)})
            

#%% test.csv oluşturma
with open ("test.csv",mode='w') as test:
    fieldnames1 = ['category', 'pixels']
    test_writer = csv.DictWriter(test,fieldnames=fieldnames1)
    test_writer.writeheader()
    for r in rakamlar_test:
        resim_isimleri_test = os.listdir(str(os.getcwd()) + "/mnist-test/" + r)
        for i in resim_isimleri_test:
            a = Image.open(str(os.getcwd()) + "/mnist-test/" + str(r) +"/"+ str(i))
            resim = np.array(a)
            resim = resim.flatten()
            test_writer.writerow({'category': str(r),'pixels': str(resim)})
                 



