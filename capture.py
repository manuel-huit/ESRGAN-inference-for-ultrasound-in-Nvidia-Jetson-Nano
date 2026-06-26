1  #import CV
2  import numpy as np
3  import cv2 as cv
4  
5  #import torch and model
6  from models import GeneratorRRDB
7  from datasets import denormalize, mean, std
8  import torch
9  from torch.autograd import Variable
10 import argparse
11 import os
12 from torchvision import transforms
13 from torchvision.utils import save_image
14 from PIL import Image
15 
16 #import time
17 import time
18 
19 
20 '''mouse click events'''
21 #define limits of the screen size
22 xSizeScreen = 720
23 ySizeScreen = 576
24 ROI_size_x = 241
25 ROI_size_y = 241
26 #set global variable to store events to 0
27 evt = 0
28 def mouseClick(event, xPos, yPos, flags, params):
29   global pnt1
30   global evt
31   if event == cv.EVENT_LBUTTONUP:
32     print(event)
33     pnt1=(xPos, yPos)
34     evt=event
35 
36 
37 '''pytorch code'''
38 #prepare the model and check CUDA
39 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
40 
41 #Define model and load model checkpoint
42 checkpoint_model = "./saved_models/generator_94.pth"
43 generator = GeneratorRRDB(3, filters=64, num_res_blocks=23).to(device)
44 generator.load_state_dict(torch.load(checkpoint_model))
45 generator.eval()
46 
47 transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
48 
49 
50 '''opencv code '''
51 cap = cv.VideoCapture(0)
52 if not cap.isOpened():
53   print("can not open the capture")
54   exit()
55 
56 cv.namedWindow('frame')
57 cv.setMouseCallback('frame', mouseClick)
58 fn=0
59 
60 
61 
62 while True:
63   ret, frame = cap.read()
64 
65   if not ret:
66     print("can not receive frame")
67     break
68 
69   #first button
70   button_upper_coord=(0,0)
71   button_bottom_coord=(70,50)
72   cv.rectangle(frame, button_upper_coord, button_bottom_coord ,(0,0,240),2)
73 
74  
75   #second button
76   button_upper_coord=(80,0)
77   button_bottom_coord=(150,50)
78   cv.rectangle(frame, button_upper_coord, button_bottom_coord ,(240,0,0),2) 
79 
80 
81   #region of capture
82   button_upper_coord=(400,200)
83   button_bottom_coord=(641,441)
84   cv.rectangle(frame, button_upper_coord, button_bottom_coord ,(0,240,0),2) 
85 
86   
87   #quit
88   if cv.waitKey(1) == ord('q'):
89     break
90 
91   #original = 4
92   if evt==4:
93 
94     print(pnt1[0])
95     print(pnt1[1])
96       
97     if ( ((pnt1[0] >  80) and (pnt1[0] <  150)) and ((pnt1[1] >  0)  and    (pnt1[1] <  50)) ):
98       #in this case for frame first is y and then x
99       ROI = frame[200:441, 400:641]      
100
101      #ROI to super resolution
102      image_tensor = Variable(transform(ROI)).to(device).unsqueeze(0)  
103
104      #Upsample image
105      with torch.no_grad():
106        sr_image = denormalize(generator(image_tensor)).cpu()
107
108      #save_image(sr_image, "./images/outputs/esrgan-1.png")
109      fn = fn+1
110      string1 = str(fn)
111      save_image(sr_image, f"images/outputs/esrgan-{string1}.png")
112      time.sleep(7)
113
114      evt=0
115   
116  if evt==4:
117    if ( ((pnt1[0] >  0) and (pnt1[0] <  70)) and ((pnt1[1] >  0)  and    (pnt1[1] <  50)) ):
118
119     
120      # Prepare input
121      image_path="./images/inputs/bilineal-IMG_2023.jpg"
122      image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)
123
124      # Upsample image
125      with torch.no_grad():
126        sr_image = denormalize(generator(image_tensor)).cpu()
127
128      # Save image
129      fn = image_path.split("/")[-1]
130      save_image(sr_image, f"images/outputs/esrgan-{fn}")
131
132      evt=0
133
134
135  
136  cv.imshow('frame',frame)
137
138cap.release()
139cv.destroyAllWindows()
140