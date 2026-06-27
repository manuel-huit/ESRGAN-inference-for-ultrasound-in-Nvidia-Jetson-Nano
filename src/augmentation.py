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
29        # Aplica la secuencia de aumentación aleatoria a la imagen original
30        augmented_image = seq.augment_image(original_image)
31
32        # Guarda la imagen aumentada
33        output_path = os.path.join(output_folder, f"{file_name}_aug_{i}{file_extension}")
34        cv2.imwrite(output_path, augmented_image)
35
36def iterate_over_images(directory_path, output_folder, num_augmentations=2):
37    # Itera sobre archivos en el directorio
38    for filename in os.listdir(directory_path):
39        if filename.endswith(".jpg") or filename.endswith(".png"):
40            # Construye la ruta completa de la imagen
41            image_path = os.path.join(directory_path, filename)
42
43            # Aplica data augmentation a la imagen
44            apply_random_data_augmentation(image_path, output_folder, num_augmentations)
45
46if __name__ == "__main__":
47    # Ruta del directorio de imágenes originales
48    input_directory = "./Butte/"
49
50    # Carpeta de salida para las imágenes aumentadas
51    output_folder = "./Output/"
52
53    # Número de aumentaciones a aplicar
54    num_augmentations = 2
55
56    # Itera sobre imágenes y aplica data augmentation
57    iterate_over_images(input_directory, output_folder, num_augmentations)
