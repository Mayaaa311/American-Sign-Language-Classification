import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import os
from PIL import Image
import csv
import cv2 
import numpy as np 

def flip_and_rotation(directory):
    # reference: https://www.analyticsvidhya.com/blog/2022/04/image-augmentation-using-3-python-libraries/
    file_list = os.listdir(directory)
    IsGenerate = False
    for filename in file_list:
        check_flip_name = filename.replace(".png", "_flip.png")
        check_rot_name = filename.replace(".png", "_rot.png")
        if ((check_flip_name not in file_list) and ("flip" not in filename) and 
            (check_rot_name not in file_list) and ("rot" not in filename) and ("edge" not in filename)):
            IsGenerate = True
            f = os.path.join(directory, filename)
            input_img = imageio.imread(f)
            # Horizontal Flip
            hflip= iaa.Fliplr(p=1.0)
            input_hf= hflip.augment_image(input_img)
            new_file_name_flip = f.replace(".png", "_flip.png")
            imageio.imwrite(new_file_name_flip, input_hf)
            # Rotation
            rot1 = iaa.Affine(rotate=(-50,20))
            input_rot1 = rot1.augment_image(input_img)
            new_file_name_rot = f.replace(".png", "_rot.png")
            imageio.imwrite(new_file_name_rot, input_rot1)

    # Write to output.csv
    if IsGenerate:
        new_rows = []
        with open('output.csv', newline='') as input_file:
            reader = csv.reader(input_file)
            column_name = next(reader)
            start_line = 2
            for row in reader:
                if start_line > 167:
                    break
                row = row[0].split(';')
                if row[10] == "train":
                    new_row_flip = row.copy()
                    new_row_flip[1] = new_row_flip[1]+"_flip"
                    new_rows.append(new_row_flip)
                    new_row_rot = row.copy()
                    new_row_rot[1] = new_row_rot[1]+"_rot"
                    new_rows.append(new_row_rot)
                start_line += 1

        #print(new_rows)
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter =";")
            for row in new_rows:
                writer.writerow(row)
            csvfile.close()


def adaptive_threshold(directory):
    # reference: https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-2-adaptive-thresholding/
    file_list = os.listdir(directory)
    IsGenerate = False
    for filename in file_list:
        check_thresh_mean_name = filename.replace(".png", "_thresh_mean.png")
        check_thresh_gaussian_name = filename.replace(".png", "_thresh_gaussian.png")
        if ((check_thresh_mean_name not in file_list) and ("thresh_mean" not in filename) and 
            (check_thresh_gaussian_name not in file_list) and ("thresh_gaussian" not in filename) and 
            ("flip" not in filename) and ("rot" not in filename) and ("edge" not in filename)):
            IsGenerate = True
            f = os.path.join(directory, filename)
            image = cv2.imread(f) 
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            adaptive_image1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
            adaptive_image2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
            new_file_name_thresh_mean = f.replace(".png", "_thresh_mean.png")
            new_file_name_thresh_gaussian = f.replace(".png", "_thresh_gaussian.png")
            cv2.imwrite(new_file_name_thresh_mean, adaptive_image1)
            cv2.imwrite(new_file_name_thresh_gaussian, adaptive_image2)
    
    # Write to output.csv
    if IsGenerate:
        new_rows = []
        with open('output.csv', newline='') as input_file:
            reader = csv.reader(input_file)
            column_name = next(reader)
            start_line = 2
            for row in reader:
                if start_line > 167:
                    break
                row = row[0].split(';')
                if row[10] == "train":
                    new_row_flip = row.copy()
                    new_row_flip[1] = new_row_flip[1]+"_thresh_mean"
                    new_rows.append(new_row_flip)
                    new_row_rot = row.copy()
                    new_row_rot[1] = new_row_rot[1]+"_thresh_gaussian"
                    new_rows.append(new_row_rot)
                start_line += 1

        #print(new_rows)
        with open('output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter =";")
            for row in new_rows:
                writer.writerow(row)
            csvfile.close()

    
def edge_detection(diretory):
    # reference: https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
    file_list = os.listdir(diretory)
    for filename in file_list:
        check_edge_name = filename.replace(".png", "_edge.png")
        if (check_edge_name not in file_list) and ("edge" not in filename) and ("flip" not in filename and ("rot" not in filename)):
            f = os.path.join(diretory, filename)
            input_img = cv2.imread(f, 0)
            edges = cv2.Canny(input_img, 100, 200)
            new_file_name_edge = f.replace(".png", "_edge.png")
            cv2.imwrite(new_file_name_edge, edges)

    # Write to output.csv
    new_rows = []
    with open('output.csv', newline='') as input_file:
        reader = csv.reader(input_file)
        column_name = next(reader)
        for row in reader:
            row = row[0].split(';')
            if row[10] == "train":
                new_row_edge = row.copy()
                new_row_edge[1] = new_row_edge[1]+"_edge"
                new_rows.append(new_row_edge)

    #print(new_rows)
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter =";")
        for row in new_rows:
            writer.writerow(row)
        csvfile.close()

def main():
    # generate flipped and rotated images
    print("generating data_augmentation method...")

if __name__ == "__main__":
    main()