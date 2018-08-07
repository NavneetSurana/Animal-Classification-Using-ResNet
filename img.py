import os,sys
import shutil
import pandas as pd

data=pd.read_csv('D:/MachineLearning/AnimalClassification/train.csv')
Im_id=data['Image_id']
Animal=data['Animal']
dic_data=dict()
for i in range(0,len(Im_id)):
	dic_data[Im_id[i].strip()]=Animal[i].strip()

source_dir='D:/MachineLearning/AnimalClassification/Images/train'
folder='D:/MachineLearning/AnimalClassificationUsingCNN'
lis=os.listdir(source_dir)
j=0
for filename in lis:
	src_filename=os.path.join(source_dir,filename)
	temp=os.path.join(folder,'Images/test/'+dic_data[filename])
	if j<9000:
		temp=os.path.join(folder,'Images/train/'+dic_data[filename])
	if not os.path.exists(temp):
		os.makedirs(temp)
	dst_filename=os.path.join(temp,filename)

	if os.path.isfile(src_filename) and not os.path.isfile(dst_filename):
	 	shutil.copy(src_filename,temp)
	j=j+1;



lis=os.listdir(folder+'/Images/val');

for i in lis:
	temp=folder+'/Images/train/'+i.strip();
	temp1=folder+'/Images/val/'+i.strip();
	if not os.path.exists(temp1):
		os.makedirs(temp1)
	k=os.listdir(temp1)
	for j in range(0,int(len(k))):
		shutil.move(temp1+'/'+k[j],temp+'/'+k[j])


lis=os.listdir(folder+'/Images/train')
min=None
for i in lis:
	temp=folder+'/Images/train/'+i.strip()
	temp=os.listdir(temp)
	print(len(temp))