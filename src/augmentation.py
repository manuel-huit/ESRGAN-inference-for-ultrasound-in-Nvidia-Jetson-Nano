
"""
Script: augmentation.py
Description: Seen the low quantity of samples for training (ultrasound images), the batch was increased by applying
             data augmentation. This script reads a folder with samples and then applies data augmentaion by transformig
             the images with rotation, scaling, blur, saturation

Author: Manuel Huitrado
Date: June 2026 (originaly created in 2024)
Versión: 1.0.0
Licence: MIT License (as origanly created by eriklindernoren )

Dependencies:
- opencv
- os
- time
- imgaug
"""





1 import os
2 import cv2
3 import imgaug as ia
4 from imgaug import augmenters as iaa
5 
6 def apply_random_data_augmentation(image_path, output_folder, num_augmentations=2):
7     # Crea el directorio de salida si no existe
8     if not os.path.exists(output_folder):
9         os.makedirs(output_folder)
10
11    # Lee la imagen original
12    original_image = cv2.imread(image_path)
13
14    # Obtiene el nombre y la extensión del archivo
15    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
16
17    # Define una secuencia de aumentación aleatoria
18    seq = iaa.Sequential([
19        iaa.SomeOf((1, 3), [
20            iaa.Affine(rotate=(-45, 45)),
21            iaa.Multiply((0.5, 1.5)),
22            iaa.MultiplySaturation((0.5, 1.5)),
23            iaa.GaussianBlur(sigma=(0.0, 3.0)),
24            iaa.Affine(scale=(0.5, 1.5))
25        ])
26    ], random_order=True)
27
28    for i in range(num_augmentations):
29        # random transformation
30        augmented_image = seq.augment_image(original_image)
31
32        # stores the image with following format     filename_aug_nameofile.png
33        output_path = os.path.join(output_folder, f"{file_name}_aug_{i}{file_extension}")
34        cv2.imwrite(output_path, augmented_image)
35
36def iterate_over_images(directory_path, output_folder, num_augmentations=2):
37    # iteration over folder
38    for filename in os.listdir(directory_path):
39        if filename.endswith(".jpg") or filename.endswith(".png"):
40            #
41            image_path = os.path.join(directory_path, filename)
42
43            # data augmentation
44            apply_random_data_augmentation(image_path, output_folder, num_augmentations)
45
46if __name__ == "__main__":
47    # path for original images
48    input_directory = "./Butte/"
49
50    # path for output
51    output_folder = "./Output/"
52
53    # types of transformations to be applied
54    num_augmentations = 2
55
56    # iteration function
57    iterate_over_images(input_directory, output_folder, num_augmentations)
