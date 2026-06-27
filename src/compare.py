1

"""
Script: compare.py
Description: Script aimed to compare 2 images: the original high resolution image vs low resolution images upscaled by ESRGAN. 
             The idea is to calculate PNSR and SSIM over a region of interest. This is helpful to measure the performance
             of the ESRGAN against other methods of super resolution.

Author: Manuel Huitrado
Date: June 2026 (originaly created in 2024)
Versión: 1.0.0
Licence: MIT License (as origanly created by eriklindernoren )

Dependencies:
- opencv
- math
- numpy
- argparse
- os
- time
- glob
- the SSIM and PSNR algorithms were taken from missing website links
"""

import math
2  import numpy as np
3  import cv2
4  import argparse
5  import time, math, glob
6  import os
7  
8  parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
9  parser.add_argument("--dataset", default="C:/Users/M/PycharmProjects/pythonProject2/venv/Set5", type=str, help="dataset name, Default: Set5")
10 
11 
12 
13 def calculate_psnr(img1, img2, x_mask, y_mask):
14     # img1 and img2 have range [0, 255]
15     image1 = cv2.imread(img1)
16     image2 = cv2.imread(img2)
17 
18     #mask creation
19     mask = np.zeros(image1.shape, dtype=np.uint8)
20     # cv2.circle(image, center_coordinates, radius, color, thickness)
21     mask = cv2.circle(mask, (x_mask, y_mask), 125, (255, 255, 255), -1)
22 
23     # Mask input image with binary mask image 1
24     result1 = cv2.bitwise_and(image1, mask)
25     # Color background white
26     result1[mask == 0] = 255  # Optional
27 
28     # Mask input image with binary mask image 2
29     result2 = cv2.bitwise_and(image2, mask)
30     # Color background white
31     result2[mask == 0] = 255  # Optional
32 
33     cv2.imshow('completa', image1)
34     cv2.imshow('original', result1)
35     cv2.imshow('result', result2)
36     cv2.waitKey()
37 
38 
39     calculation1 = result1.astype(np.float64)
40     calculation2 = result2.astype(np.float64)
41     mse = np.mean((calculation1 - calculation2)**2)
42     if mse == 0:
43         return float('inf')
44     return 20 * math.log10(255.0 / math.sqrt(mse))
45 
46 
47 
48 def ssim(img1, img2):
49 
50     C1 = (0.01 * 255)**2
51     C2 = (0.03 * 255)**2
52 
53     img1 = img1.astype(np.float64)
54     img2 = img2.astype(np.float64)
55     kernel = cv2.getGaussianKernel(11, 1.5)
56     window = np.outer(kernel, kernel.transpose())
57 
58     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
59     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
60     mu1_sq = mu1**2
61     mu2_sq = mu2**2
62     mu1_mu2 = mu1 * mu2
63     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
64     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
65     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
66 
67     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
68                                                             (sigma1_sq + sigma2_sq + C2))
69     return ssim_map.mean()
70 
71 
72 def calculate_ssim(img1, img2, x_mask, y_mask):
73     '''calculate SSIM
74     the same outputs as MATLAB's
75     img1, img2: [0, 255]
76     '''
77     img1 = cv2.imread(img1)
78     img2 = cv2.imread(img2)
79 
80 
81     #mask creation
82     mask = np.zeros(img1.shape, dtype=np.uint8)
83     # cv2.circle(image, center_coordinates, radius, color, thickness)
84     mask = cv2.circle(mask, (x_mask, y_mask), 125, (255, 255, 255), -1)
85 
86     # Mask input image with binary mask image 1
87     result1 = cv2.bitwise_and(img1, mask)
88     # Color background white
89     result1[mask == 0] = 255  # Optional
90 
91     # Mask input image with binary mask image 2
92     result2= cv2.bitwise_and(img2, mask)
93     # Color background white
94     result2[mask == 0] = 255  # Optional
95 
96 
97     if not result1.shape == result2.shape:
98         raise ValueError('Input images must have the same dimensions.')
99     if result1.ndim == 2:
100        return ssim(result1, result2)
101    elif result1.ndim == 3:
102        if result1.shape[2] == 3:
103            ssims = []
104            for i in range(3):
105                ssims.append(ssim(result1, result2))
106            return np.array(ssims).mean()
107        elif result1.shape[2] == 1:
108            return ssim(np.squeeze(result1), np.squeeze(result2))
109    else:
110        raise ValueError('Wrong input image dimensions.')
111
112
113
114
115opt = parser.parse_args()
116image_list = glob.glob(opt.dataset+"/*.*")
117
118
119#Captura1
120x_mask = 580
121y_mask = 200
122print ("PSNR")
123#image 1 Original   image2 SR
124print (calculate_psnr('./Original/Captura1.png','./UpScaled_Bilinear/bilineal-bilineal-Captura1.png', x_mask, y_mask))
125#print (calculate_psnr('./Original/Captura1_normalizado.png','./UpScaled_Bicubic/bicubic-bilineal-Captura1_normalizado.png', x_mask, y_mask))
126#print (calculate_psnr('./Original/Captura1_normalizado.png','./UpScaled_ESRGAN/esrgan-bilineal-Captura1.png', x_mask, y_mask))
127#print (calculate_psnr('./Original/Captura1_normalizado.png','./UpScaled_SRGAN/srgan-bilineal-Captura1_normalizado.png', x_mask, y_mask))
128
129print ("SSIM")
130print (calculate_ssim('./Original/Captura1.png','./UpScaled_SRGAN/srgan-bilineal-Captura1.png', x_mask, y_mask))
131#print (calculate_ssim('./Original/Captura1_normalizado.png','./UpScaled_Bicubic/bicubic-bilineal-Captura1_normalizado.png', x_mask, y_mask))
132#print (calculate_ssim('./Original/Captura1_normalizado.png','./UpScaled_ESRGAN/esrgan-bilineal-Captura1.png', x_mask, y_mask))
133#print (calculate_ssim('./Original/Captura1_normalizado.png','./UpScaled_SRGAN/srgan-bilineal-Captura1_normalizado.png', x_mask, y_mask))
134
135
136'''
137#Captura2
138x_mask = 380
139y_mask = 400
140print ("PSNR")
141#image 1 Original   image2 SR
142print (calculate_psnr('./Original/Captura2.png','./UpScaled_Bilinear/bilineal-bilineal-Captura2.png', x_mask, y_mask))
143#print (calculate_psnr('./Original/Captura2.png','./UpScaled_Bicubic/bicubic-bilineal-Captura2.png', x_mask, y_mask))
144#print (calculate_psnr('./Original/Captura2.png','./UpScaled_ESRGAN/esrgan-bilineal-Captura2_normalizado.png', x_mask, y_mask))
145#print (calculate_psnr('./Original/Captura2.png','./UpScaled_SRGAN/srgan-bilineal-Captura2.png', x_mask, y_mask))
146
147print ("SSIM")
148print (calculate_ssim('./Original/Captura2.png','./UpScaled_SRGAN/srgan-bilineal-Captura2.png', x_mask, y_mask))
149#print (calculate_ssim('./Original/Captura2.png','./UpScaled_Bicubic/bicubic-bilineal-Captura2.png', x_mask, y_mask))
150#print (calculate_ssim('./Original/Captura2.png','./UpScaled_ESRGAN/esrgan-bilineal-Captura2_normalizado.png', x_mask, y_mask))
151#print (calculate_ssim('./Original/Captura2.png','./UpScaled_SRGAN/srgan-bilineal-Captura2.png', x_mask, y_mask))
152'''
153
154
.....