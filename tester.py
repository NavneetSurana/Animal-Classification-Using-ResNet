import pickle
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input 
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50


lis=np.array([ i.strip()  for i in pd.read_csv('test.csv')['Image_id']])

liss=['Images/test/'+i for i in lis]

# lis =np.array(os.listdir('Images/val/antelope'))
# liss=['Images/val/antelope/'+i.strip() for i in lis]

import json
json_file=open('model_arch.json','r')
model_arc_data=json_file.read()
json_file.close()

from tensorflow.python.keras.models import model_from_json
my_new_model = model_from_json(model_arc_data)
my_new_model.load_weights('my_model_weights.h5')
my_new_model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

probs=np.zeros((len(lis),30))
jj=0
pic_in=open('animal_dict','rb')
anim_dic=pickle.load(pic_in)
pic_in.close()

lims=sorted(anim_dic.keys())
for i in liss:
	imgs=load_img(i,target_size=(224,224))
	img_arr=np.array([img_to_array(imgs)])
	test_data=preprocess_input(img_arr)
	preds=my_new_model.predict(test_data)
	probs[jj]=preds[0]
	# preds=my_new_model.predict_classes(test_data)
	# print(lims[preds[0]],preds[0])
	jj+=1
	print(jj)

pic_in=open('probs','wb')
pickle.dump(probs,pic_in)
pic_in.close()
lis=lis.reshape(len(lis),1)
Final_set=np.concatenate((lis,probs),axis=1)

pic_in=open('animal_dict','rb')
anim_dic=pickle.load(pic_in)
pic_in.close()

lis=sorted(anim_dic.keys())
header=['image_id']
for i in lis:
	header.append(i.strip())
print(header)
Frame=pd.DataFrame(Final_set,columns=header)
Frame.to_csv('test_result.csv',index=False)










