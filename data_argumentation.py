import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import os
from PIL import Image

def data_argumentation(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        input_img = imageio.imread(f)
        #Horizontal Flip
        hflip= iaa.Fliplr(p=1.0)
        input_hf= hflip.augment_image(input_img)
        new_file_name_flip = f.replace(".png", "_flip.png")
        #Rotation
        imageio.imwrite(new_file_name_flip, input_hf)
        rot1 = iaa.Affine(rotate=(-50,20))
        input_rot1 = rot1.augment_image(input_img)
        new_file_name_rot = f.replace(".png", "_rot.png")
        imageio.imwrite(new_file_name_rot, input_rot1)

def main():
    # data argumentation
    data_argumentation("avg_train")

if __name__ == "__main__":
    main()