import numpy as np
import sys
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


file_path = sys.argv[1]
#-------------PRETRAINED MODEL FEATURE EXTRACTION---------------------

model = VGG16(weights='imagenet', include_top=False)
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

feature_list_np = np.array(feature_list)

#-----------------DIMENSIONALITY REDUCTION-----------------------------

pca = joblib.load('pca_features.pkl') 
X_std = StandardScaler().fit_transform(feature_list_np)
pca_features = pca.transform(X_std)
print("Pca features shape", pca_features.shape)

#--------------K-Nearest Neighbors------------------------------------


file_list = np.load("file_list_grey.npy")
neigh = joblib.load('neigh.pkl')
result=neigh.radius_neighbors(pca_features, 3)
duplicates = np.array((file_list(result)))

print("Duplicate files are", duplicates)




