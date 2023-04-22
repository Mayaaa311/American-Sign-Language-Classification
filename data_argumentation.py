import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import os
from PIL import Image
import csv

def flip_and_Rotation(directory):
    # reference: https://www.analyticsvidhya.com/blog/2022/04/image-augmentation-using-3-python-libraries/
    file_list = os.listdir(directory)
    for filename in file_list:
        check_flip_name = filename.replace(".png", "_flip.png")
        check_rot_name = filename.replace(".png", "_rot.png")
        if ((check_flip_name not in file_list) and ("flip" not in filename) and 
            (check_rot_name not in file_list) and ("rot" not in filename)):
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
    new_rows = []
    with open('output.csv', newline='') as input_file:
        reader = csv.reader(input_file)
        column_name = next(reader)
        for row in reader:
            row = row[0].split(';')
            if row[10] == "train":
                new_row_flip = row.copy()
                new_row_flip[1] = new_row_flip[1]+"_flip"
                new_rows.append(new_row_flip)
                new_row_rot = row.copy()
                new_row_rot[1] = new_row_rot[1]+"_rot"
                new_rows.append(new_row_rot)

    #print(new_rows)
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter =";")
        for row in new_rows:
            writer.writerow(row)
        csvfile.close()


def main():
    # generate flipped and rotated images
    flip_and_Rotation("avg_train")

if __name__ == "__main__":
    main()