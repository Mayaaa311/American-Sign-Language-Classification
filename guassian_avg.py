from PIL import Image, ImageFilter
import csv
import os
from os import listdir
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset = "ChicagoFSWild/"
os.mkdir("avg_train")
#f = open('train_id_label.csv', 'w')
#writer_f = csv.writer(f)
with open('output_dev.csv', newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=';')
    column_name = next(reader)
    #id = 0
    for row in reader:
        # get label
        label = row[12]
        # get all photos under this folder
        folder_name = row[1]
        absolute_folder_name = dataset+"/"+folder_name
        Gaussian_list = []
        for image in os.listdir(absolute_folder_name):
            # for each image
            # apply guassian filter
            Gaussian_image = image.filter(ImageFilter.GaussianBlur)
            Gaussian_list.append(Gaussian_image)
        # get avg and produce a new avg photo; named it and stored it in train folder
        # reference: https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/
        avg_image = Gaussian_list[0]
        for i in range(len(Gaussian_list)):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(Gaussian_list[i], alpha, avg_image, beta, 0.0)
        new_file_name = "avg_train/"+label+".png"
        cv2.imwrite(new_file_name, avg_image)
        #avg_image = cv2.imread(new_file_name)
        #plt.imshow(avg_image)
        #plt.show()
        
# training model
# reference: https://www.kaggle.com/code/kattat/dogs-vs-cats-image-classification-with-cnn
# create folders and move files
train_dir = 'avg_train'
if not os.path.exists((os.path.join(train_dir, 'a'))):
    os.mkdir(os.path.join(train_dir, 'a'))
if not os.path.exists((os.path.join(train_dir, 'b'))):
    os.mkdir(os.path.join(train_dir, 'b'))
if not os.path.exists((os.path.join(train_dir, 'c'))):
    os.mkdir(os.path.join(train_dir, 'c'))
if not os.path.exists((os.path.join(train_dir, 'd'))):
    os.mkdir(os.path.join(train_dir, 'd'))

for file in os.listdir(train_dir):
    if file[0] == 'a':
        os.replace(os.path.join(train_dir,file), os.path.join(train_dir,'a', file))
    elif file[0] == 'b':
        os.replace(os.path.join(train_dir,file), os.path.join(train_dir,'b', file))
    elif file[0] == 'c':
        os.replace(os.path.join(train_dir,file), os.path.join(train_dir,'c', file))
    else:
        os.replace(os.path.join(train_dir,file), os.path.join(train_dir,'d', file))

BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

image_generator = ImageDataGenerator(rescale=1./255, validation_split = 0.2)

train_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE), # (150,150)
                                                     subset='training')

val_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=train_dir,
                                                   shuffle=False,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE), # (150,150)
                                                   subset='validation')


    