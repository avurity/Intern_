#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Mmachad/autoencoder/blob/master/OCR_Test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


from google.colab import drive
drive.mount('/content/Drive')


# In[ ]:


import cv2
from google.colab.patches import cv2_imshow
import keras
admodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel200.h5')


# In[ ]:


get_ipython().system('sudo apt install tesseract-ocr')
get_ipython().system('pip install pytesseract')


# In[ ]:





# In[174]:


from keras.preprocessing.image import load_img, array_to_img, img_to_array
import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image


# In[196]:


from pathlib import Path
input_dir3  = Path('/content/Drive/My Drive/autoencoderData/Model_test_Images')
test = input_dir3 / 'adTest'


# In[197]:


import numpy as np
test_images = sorted(os.listdir(test))
sample_test = load_img(test/ test_images[15], target_size=(1000,504))
sample_test = img_to_array(sample_test)
sample_test_img = sample_test.astype('float32')/255.
sample_test_img = np.expand_dims(sample_test, axis=0)


# In[177]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# In[200]:


import matplotlib.pyplot as plt
admodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/AD10000encoded_decoded_weights.h5')
test_images = sorted(os.listdir(test))
sample_test = load_img(test/ test_images[18], target_size=(1000,504))
 #sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
sample_test1 = img_to_array(sample_test)
sample_test_img = sample_test1/255
sample_test_img = np.expand_dims(sample_test_img, axis=0)
predicted_label = admodel.predict(sample_test_img)
f, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(sample_test)
ax[1].imshow(np.squeeze(predicted_label))
plt.show()


# In[201]:


import cv2
predicted_images = []
for i in range(0,30):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[0].title.set_text('Actual Image')
  ax[1].imshow(np.squeeze(predicted_label))
  ax[1].title.set_text('Predicted Image')
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  #print("####################################################################                                                         ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img.astype(np.uint8))
  #up_image = cv2.resize(predicted_la,(1000,700),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  #t = pytesseract.image_to_string(up_image)
  extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--oem 3 --psm 6')
  print(extractedInformation)


# ### TEST **CODE**

# In[171]:


import cv2
for i in range(0,20):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img.astype(np.uint8))
  #up_image = cv2.resize(predicted_la,(1000,700),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  #t = pytesseract.image_to_string(up_image)
  extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--psm 6')
  print(extractedInformation)


# # Ad Model _10000

# In[183]:


import matplotlib.pyplot as plt
admodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel10000.h5')
test_images = sorted(os.listdir(test))
sample_test = load_img(test/ test_images[18], target_size=(1000,504))
 #sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
sample_test1 = img_to_array(sample_test)
sample_test_img = sample_test1/255
sample_test_img = np.expand_dims(sample_test_img, axis=0)
predicted_label = admodel.predict(sample_test_img)
f, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(sample_test)
ax[1].imshow(np.squeeze(predicted_label))
plt.show()


# In[185]:


import cv2
predicted_images = []
for i in range(0,21):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  predicted_images.append(predicted_label)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[0].title.set_text('Actual Image')
  ax[1].imshow(np.squeeze(predicted_label))
  ax[1].title.set_text('Predicted Image')
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   ")
  print("  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img.astype(np.uint8))
  #up_image = cv2.resize(predicted_la,(1000,700),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  #t = pytesseract.image_to_string(up_image)
  extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--oem 3 --psm 6')
  print(extractedInformation)


# # **AD_model_3K**

# In[186]:


import matplotlib.pyplot as plt
admodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel3k_512LD_max.h5')
test_images = sorted(os.listdir(test))
sample_test = load_img(test/ test_images[18], target_size=(1000,504))
 #sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
sample_test1 = img_to_array(sample_test)
sample_test_img = sample_test1/255
sample_test_img = np.expand_dims(sample_test_img, axis=0)
predicted_label = admodel.predict(sample_test_img)
f, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(sample_test)
ax[1].imshow(np.squeeze(predicted_label))
plt.show()


# In[188]:


import cv2
predicted_images = []
for i in range(0,19):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  predicted_images.append(predicted_label)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[0].title.set_text('Actual Image')
  ax[1].imshow(np.squeeze(predicted_label))
  ax[1].title.set_text('Predicted Image')
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  #print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                        ")
  #print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                        ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  #up_image = cv2.resize(predicted_label,(1000,570),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img.astype(np.uint8))
  #t = pytesseract.image_to_string(up_image)
  extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--oem 3 --psm 6')
  print(extractedInformation)


# In[152]:


import cv2
predicted_images = []
for i in range(10,5):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  predicted_images.append(predicted_label)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[0].title.set_text('Actual Image')
  ax[1].imshow(np.squeeze(predicted_label))
  ax[1].title.set_text('Predicted Image')
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  #print(" ----------------------                                                        ")
  #print(" -----------------------                                                        ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = np.squeeze(admodel.predict(sample_test_img))
  #up_image = cv2.resize(predicted_label,(1000,570),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img.astype(np.uint8))
  #t = pytesseract.image_to_string(up_image)
  extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--oem 3 --psm 6')
  print(extractedInformation)


# In[194]:


sample_test = load_img(test/ test_images[7])
extractedInformation = pytesseract.image_to_string(sample_test)
print(extractedInformation)


# **Memo - 200 Model**

# In[ ]:


import cv2
from google.colab.patches import cv2_imshow
import keras
newsmodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/memoModel200.h5')


# In[ ]:


from pathlib import Path
input_dir3  = Path('/content/Drive/My Drive/autoencoderData/memoTest')
test = input_dir3 / 'memoTest'


# In[ ]:


import numpy as np
test_images = sorted(os.listdir(test))


# In[ ]:


for i in range(100,115):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = newsmodel.predict(sample_test_img)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[1].imshow(np.squeeze(predicted_label))
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  print("                                                         ")
  print("                                                         ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  sample_test = img_to_array(sample_test)
  sample_test_img = sample_test.astype('float32')/255
  sample_test_img = np.expand_dims(sample_test, axis=0)
  predicted_label = np.squeeze(newsmodel.predict(sample_test_img))
  extractedInformation = pytesseract.image_to_string(predicted_label.astype('uint8'))
  print(extractedInformation)


# ## **NEWS -200 EPOCHS Model**

# In[ ]:


import cv2
from google.colab.patches import cv2_imshow
import keras
newsmodel = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/NEWS200encoded_decoded_weights.h5')


# In[ ]:


from pathlib import Path
input_dir3  = Path('/content/Drive/My Drive/autoencoderData/newsTest')
test = input_dir3 / 'Testnews'


# In[ ]:


import numpy as np
test_images = sorted(os.listdir(test))


# In[ ]:


for i in range(1,15):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = newsmodel.predict(sample_test_img)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,2, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[1].imshow(np.squeeze(predicted_label))
  plt.show()
  print("Actual OCR TEXT :")
  extractedInformation = pytesseract.image_to_string(sample_test)
  print(extractedInformation)
  print("                                                         ")
  print("                                                         ")
  print("Predicted OCR TEXT :")
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  sample_test = img_to_array(sample_test)
  sample_test_img = sample_test.astype('float32')/255
  sample_test_img = np.expand_dims(sample_test, axis=0)
  predicted_label = np.squeeze(newsmodel.predict(sample_test_img))
  extractedInformation = pytesseract.image_to_string(predicted_label.astype('uint8'))
  print(extractedInformation)


# In[226]:


admodel200 = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel200.h5')
admodel10000 = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel10000.h5')
admodel3000 = keras.models.load_model('/content/Drive/My Drive/autoencoderData/Working Models/adModel3k_512LD_max.h5')
for i in range(0,33):
  sample_test = load_img(test/ test_images[i],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
  sample_test1 = img_to_array(sample_test)
  sample_test_img = sample_test1/255
  sample_test_img = np.expand_dims(sample_test_img , axis=0)
  predicted_label = admodel200.predict(sample_test_img)
  predicted_label1 = admodel10000.predict(sample_test_img)
  predicted_label2 = admodel3000.predict(sample_test_img)
  print("                                                         ")
  print("                                                         ")
  print("Test Image:",i)
  f, ax = plt.subplots(1,4, figsize=(20,10))
  ax[0].imshow(np.squeeze(sample_test),cmap='gray')
  ax[0].title.set_text('Actual Image')
  ax[1].imshow(np.squeeze(predicted_label))
  ax[1].title.set_text('200Epoch model prediction')
  ax[2].imshow(np.squeeze(predicted_label2))
  ax[2].title.set_text('3000 Epoch model prediction')
  ax[3].imshow(np.squeeze(predicted_label1))
  ax[3].title.set_text('10000Epoch model prediction')
  plt.show()
  


# In[230]:


sample_test = load_img(test/ test_images[31],target_size=(1000,504))
  #cleaned_img = predict_transform(sample_test, model)
sample_test1 = img_to_array(sample_test)
sample_test_img = sample_test1/255
sample_test_img = np.expand_dims(sample_test_img , axis=0)
predicted_label = np.squeeze(admodel3000.predict(sample_test_img))
ret,img = cv2.threshold(np.array(predicted_label), 0, 255, cv2.THRESH_BINARY)
img = Image.fromarray(img.astype(np.uint8))
  #up_image = cv2.resize(predicted_la,(1000,700),interpolation=cv2.INTER_AREA)
  #img_rgb = cv2.cvtColor(up_image, cv2.COLOR_BGR2RGB)
  #t = pytesseract.image_to_string(up_image)
extractedInformation = pytesseract.image_to_string(img,lang='eng',config='--oem 3 --psm 6')
print(extractedInformation)


# In[224]:


sample_test
extractedInformation = pytesseract.image_to_string(sample_test,lang='eng',config='--oem 3 --psm 6')
print(extractedInformation)

