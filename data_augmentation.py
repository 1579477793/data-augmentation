
# coding: utf-8

# In[1]:


# 数据增强程序
# -----Type
#    |-----Type_1
#         |- 1.jpg
#         |- 2.jpg
#        ...
#    |-----Type_2
#    |-----Type_3
# 单独对某一个文件夹增强，此处对 Type_1 增强
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import numpy as np
import glob

images = glob.glob("C:/Users/sunzheng/jupyter_notebook/Type/Type_1/" + "*.jpg")
Save_dir = 'C:/Users/sunzheng/jupyter_notebook/Type/Type_2'

data_gen = ImageDataGenerator(rotation_range=10,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              shear_range=0.02,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest',
                              data_format='channels_last')

for idx, path in enumerate(images):
    img=load_img(path, target_size=(1024, 1024))
    x = img_to_array(img,data_format="channels_last") 
    x=x.reshape((1,) + x.shape)
    i = 1
    for batch in data_gen.flow(x,batch_size=1, save_to_dir=Save_dir, save_prefix='xx', save_format='jpg'):
        i += 1
        if i>10:  # 一张图增10张,控制数量。原始有x，现在数量是 (10 x)
            break

