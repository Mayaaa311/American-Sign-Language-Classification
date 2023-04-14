import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def convert_image():
    max_h = 0
    max_w = 0
    dataset = "ChicagoFSWild-Frames"
    csvs = [['output_train.csv',"avg_train"],
            ['output_dev.csv',"avg_dev"],
            ['output_test.csv',"avg_test"]]
    
    for csv_file in csvs:
        
        dir_path = csv_file[1]
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        with open(csv_file[0], newline='') as csvfile:
            reader = csv.reader(csvfile)
            column_name = next(reader)
            for row in reader:
                # get label
                label = row[12]
                # get all photos under this folder
                folder_name = row[1]
                absolute_folder_name = dataset+"/"+folder_name
                start_index = int(row[13])-1
                end_index = int(row[14])
                first = True
                images = []
                for image_name in os.listdir(absolute_folder_name)[start_index:end_index]:
                    # For each image apply guassian filter
                    full_path = absolute_folder_name + "/" + image_name
                    # read the original image with imread
                    img = cv2.imread(full_path)
                    # convert the grayscale image to a NumPy array with the same data type as the original image
                    # img = np.asarray(gray_image, dtype=image.dtype)
                    # Check if it is the first time of imread
                    if first:
                        Gaussian_sum = np.zeros(img.shape)
                        if img.shape[0] > max_h:
                            max_h = img.shape[0]
                        if img.shape[1] > max_w:
                            max_w = img.shape[1]
                        first = False
                    filtered = cv2.GaussianBlur(img, (5, 5), 0).astype(np.int8)
                    images.append(filtered)
                # num_img = end_index - start_index + 1
                Gaussian_avg = np.mean(images, axis=0).astype(np.uint8)
                # Normalize the pixel values to the range of 0 to 255
                min_val = np.min(Gaussian_avg)
                max_val = np.max(Gaussian_avg)
                normalized_image = (Gaussian_avg - min_val) / (max_val - min_val) * 255
                # Convert the pixel values to unsigned 8-bit integers
                normalized_image = normalized_image.astype(np.uint8)
                # Save the filtered image to file
                fname = folder_name.split("/")
                new_path = dir_path+"/"+fname[0]+"_"+fname[1]+'.png'
                print(new_path)
                cv2.imwrite(new_path,normalized_image)
    return (max_h, max_w)
            # get avg and produce a new avg photo; named it and stored it in train folder
            # reference: https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/
            # avg_image = Gaussian_list[0]
            # for i in range(len(Gaussian_list)):
            #     if i == 0:
            #         pass
            #     else:
            #         alpha = 1.0/(i + 1)
            #         beta = 1.0 - alpha
            #         avg_image = cv2.addWeighted(Gaussian_list[i], alpha, avg_image, beta, 0.0)
            # new_file_name = "avg_train/"+label+".png"
            # cv2.imwrite(new_file_name, avg_image)
            #avg_image = cv2.imread(new_file_name)
            #plt.imshow(avg_image)
            #plt.show()

def resize_image(max_h, max_w):
    # Option 1: use padding to max_h and max_w
    # Option 2: use resize method needs threshold
    print(0)

def model_train():
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


def main():
    # convert a series of images to one
    max_h, max_w = convert_image()
    # convert images to have same size


if __name__ == "__main__":
    main()