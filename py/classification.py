# PROJECT - CV
# Ignacio Garrido Botella & Abel Rodriguez Romero
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import os
from skimage import io
from skimage.transform import resize
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras import regularizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


#%%

#For making the NN work:
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Problem: https://medium.com/@valeryyakovlev/python-keras-hangs-on-fit-method-spyder-anaconda-8d555eeeb47e

#Source: https://blog.keras.io/building-autoencoders-in-keras.html

# parameters that you should set before running this script
filter = ['tvmonitor', 'car', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/Data/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 256    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)


# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder) #List all the files from the subfolder "Annotations"
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f)) #To get all the xml
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4]) #It stores the file name that contains one of the words of the keyword list "filter"

# step2 - build (x,y) for TRAIN/VAL (classification)
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/") #List all the files from the subfolder "ImageSets/Main/"
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f] #List all the train files
val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f] #List all the test files


def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file: #It opens the files with all the names of the files with the pictures
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y #y_temp


x_train, y_train = build_classification_dataset(train_files)
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_classification_dataset(val_files)
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))

# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)

#Randomly permutate the train/val sets:
p_train = np.random.permutation(len(x_train))
p_val = np.random.permutation(len(x_val))
x_train=x_train[p_train]
y_train=y_train[p_train]
x_val=x_val[p_val]
y_val=y_val[p_val]


#%% Add dummy variables
import data_utils

y_val_new = np.append(y_val[:,0:4], np.zeros((len(y_val),1)), axis = 1)

for i in range(len(y_val_new)):
    if sum(y_val_new[i]) == 0:
        y_val_new[i,4] = 1

y_train_new = np.append(y_train[:,0:4], np.zeros((len(y_train),1)), axis = 1)

for i in range(len(y_train_new)):
    if sum(y_train_new[i]) == 0:
        y_train_new[i,4] = 1
        
data_utils.save_pickle("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/x_train_dummy.pkl",x_train)
data_utils.save_pickle("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/y_train_dummy.pkl",y_train_new)
data_utils.save_pickle("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/x_val_dummy.pkl",x_val)
data_utils.save_pickle("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/y_val_dummy.pkl",y_val_new)        


#%% Generate new data

datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
x_train_big = x_train.reshape((1,) + x_train.shape)       
    
x = x_train_big[0,1]
x = x.reshape((1,) + x.shape)  
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely       
        
        
 
#%% Get the autoencoder from previously trained model:

path = "/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/Computer vision/Project/Models/model.h5"  
autoencoder = load_model(path)
autoencoder.summary()

#Get encoder
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_14').output)
encoder.summary()

#Get an array with the features
features_train = encoder.predict(x_train)
features_val = encoder.predict(x_val)

#%%
#%%  CCCC  L     AAAA  SSSS  SSSS  I  FFFF  I  CCCC  AAAA  TTTTT IIIII  OOOO  N  N                                                                 
#%%  C     L     A  A  S     S     I  F     I  C     A  A    T     I    O  O  NN N                                      
#%%  C     L     AAAA  SSSS  SSSS  I  FFF   I  C     AAAA    T     I    O  O  N NN                                              
#%%  C     L     A  A     S     S  I  F     I  C     A  A    T     I    O  O  N  N                                                         
#%%  CCCC  LLLL  A  A  SSSS  SSSS  I  F     I  CCCC  A  A    T   IIIII  OOOO  N  N                                                  
#%%
#%%

# Source (normal classification): https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
# Source (images classification): https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
random.seed(42)

classifier = Sequential()

#classifier.add(Flatten())
classifier.add(Flatten(input_shape=(16,16,16)))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(5, activation='sigmoid'))

classifier.summary()


classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(features_train, y_train, epochs=50, batch_size=64, validation_data=(features_val, y_val))




def prediction(pred_vector, threshold = 0.2):
    return np.int8((pred_vector==max(pred_vector)) + (pred_vector>=threshold))





#%% Generate data
from keras.preprocessing.image import ImageDataGenerator

#Data generator:
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,zoom_range=0.2, horizontal_flip=True,fill_mode='nearest')
datagen.fit(x_train)


num_new_images_tvmonitor = 3
num_new_images_car = 1
num_new_images_dog = 1
num_new_images_bird = 2
# tv_monitor
x1 = x_train[np.where(y_train[:,0] == 1)]
y1 = y_train[np.where(y_train[:,0] == 1)]
#car
x2 = x_train[np.where(y_train[:,1] == 1)]
y2 = y_train[np.where(y_train[:,1] == 1)]
#dog
x3 = x_train[np.where(y_train[:,2] == 1)]
y3 = y_train[np.where(y_train[:,2] == 1)]
#bird
x4 = x_train[np.where(y_train[:,3] == 1)]
y4 = y_train[np.where(y_train[:,3] == 1)]


# x/y_train_new contain the new images
x_train_new = x1
y_train_new = y1

for i in range(num_new_images_tvmonitor):
    for X_batch, y_batch in datagen.flow(x1,y1, batch_size=len(x_train)):
        x_train_new = np.vstack((x_train_new,X_batch))
        y_train_new = np.vstack((y_train_new,y_batch))
        break

x_train_new = np.vstack((x_train_new,x2))
y_train_new = np.vstack((y_train_new,y2))
for i in range(num_new_images_car):
    for X_batch, y_batch in datagen.flow(x2,y2, batch_size=len(x_train)):
        x_train_new = np.vstack((x_train_new,X_batch))
        y_train_new = np.vstack((y_train_new,y_batch))
        break
 
x_train_new = np.vstack((x_train_new,x3))
y_train_new = np.vstack((y_train_new,y3))
for i in range(num_new_images_dog):
    for X_batch, y_batch in datagen.flow(x3,y3, batch_size=len(x_train)):
        x_train_new = np.vstack((x_train_new,X_batch))
        y_train_new = np.vstack((y_train_new,y_batch))
        break

x_train_new = np.vstack((x_train_new,x4))
y_train_new = np.vstack((y_train_new,y4))
for i in range(num_new_images_bird):
    for X_batch, y_batch in datagen.flow(x4,y4, batch_size=len(x_train)):
        x_train_new = np.vstack((x_train_new,X_batch))
        y_train_new = np.vstack((y_train_new,y_batch))
        break


#%%
x = x_train[1:4]
x = x.astype('float32')
# define data preparation
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,zoom_range=0.2, horizontal_flip=True,fill_mode='nearest')

datagen.fit(x_train.astype('float32'))
y=np.array([1,2,3])

x_new = 0
y_new = 0

i = 0
for (X_batch, y_batch) in datagen.flow(x,y, batch_size=64):
    x_new = X_batch[:]
    y_new = y_batch
    break

plt.imshow(x_new[0])

#%% Test VGG16 for classification
from keras.applications import vgg16

vgg = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(256, 256, 3), pooling=None, classes=100)
vgg.summary()
model = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_pool').output)

#Get an array with the features
features_train = model.predict(x_train_new)
features_val = model.predict(x_val)

classifier = Sequential()
classifier.add(Flatten(input_shape=(1,1,512)))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(4, activation='softmax'))

classifier.summary()

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(features_train, y_train_new, epochs=350, batch_size=64, validation_data=(features_val, y_val))















