import pandas as pd
import numpy as np
import urllib.request
import os.path
import sys, os
import time
import glob

from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors


#-------------------------HELPER FUNCTIONS------------------------------

def check(x):
    if x in categories:
        return 1
    return 0


def downloadImage(x):
    try:
        urls = x['imageUrlStr'].split(';')
        ext = urls[0].split('.')[-1]
        name = 'images/image/'+x['productId']+'.'+ext         
        if not os.path.isfile(name): 
            urllib.request.urlretrieve(urls[2], name)
    except:
        pass

#-------------------------------------------------------------------

print("Reading File")
data = pd.read_csv('data/2oq-c1r.csv', low_memory=True)
print("Found {} samples".format(data.shape[0]))
categories = data.categories.unique()
print("Found {} Categories".format(categories.shape[0]))

subcategories = []
for i in categories:
    if(type(i) == type('str')):
        if ('Tops' in i or 'tops' in i):
            subcategories.append(i)
print("Found {} Sub-Categories".format(len(subcategories)))

data['select'] = data['categories'].apply(lambda x: check(x))
topdata = data.loc[(data['select'] == 1)]
topdata.to_csv('data/topdata.csv')
print("Number of Top samples")
print("Found {} Top samples".format(topdata.shape[0]))

#---------------------DOWNLOADING IMAGES------------------------------

print("Downloading Only 150000 Images")
print("Downloading Images")
sample = topdata
pool = Pool(processes=100)
rows = sample.to_dict(orient='records')
for _ in tqdm(pool.imap_unordered(downloadImage, rows)):
    pass
pool.close() 
pool.join()
print("Done Downloading Images")


#-------------PRETRAINED MODEL FEATURE EXTRACTION---------------------

model = VGG16(weights='imagenet', include_top=False)
feature_list = []
file_list = []
DIR = "images/image/*"
files = glob.glob(DIR)
print("Number of files", len(files))
print("Pretrained Greyscale")

for file in tqdm(files):
    try:
        img = image.load_img(file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        VGG16_feature = model.predict(img_data)
        VGG16_feature_np = np.array(VGG16_feature)
        file_list.append(file)
        feature_list.append(VGG16_feature_np.flatten())
    except:
        pass

VGG16_feature_list_np = np.array(feature_list)
np.save("file_list_grey", file_list , allow_pickle=True)
np.save("feature_list_grey", feature_list , allow_pickle=True)

#-----------------DIMENSIONALITY REDUCTION-----------------------------

features = np.load("feature_list.npy")
print("feature shape",features.shape)
X_std = StandardScaler().fit_transform(features)
print("Starting PCA")
pca = PCA (n_components=7400)
pca.fit(X_std)

print("Explained Ratio", pca.explained_variance_ratio_)

pca_features = pca.transform(features)
print("Pca features shape", pca_features.shape)
joblib.dump(pca, 'pca.pkl') 
np.save("pca_features", pca_features, allow_pickle=True)


#--------------K-Nearest Neighbors------------------------------------

neigh = NearestNeighbors(n_neighbors=12, radius=.4,n_jobs=-1)
neigh.fit(pca_features)
joblib.dump(neigh, 'neigh.pkl')
submision = pd.Dataframe(columns = ("file","duplicate_files"))

for i in range(len(files)):
    result=neigh.radius_neighbors(files(i), 3)
    duplicates = np.array((file_list(result)))
    submision = submision.append([files(i), duplicates])

submisionpd.to_json(orient='table')




