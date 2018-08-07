import h5py
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten, GlobalAveragePooling2D
import pickle
import PIL
import scipy
weights_path = 'Input/Resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
num_classes = 30

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False,pooling = 'avg' ,weights=weights_path))
my_new_model.add(Dense(num_classes,activation='softmax'))
my_new_model.layers[0].trainable=False
print('done')

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
image_size =224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
					'Images/train',
					target_size=(image_size,image_size),
					batch_size=10,
					class_mode='categorical')
validation_generator = data_generator.flow_from_directory(
					'Images/val',
					target_size=(image_size,image_size),
					batch_size=10,
					class_mode='categorical')



my_new_model.compile(optimizer='sgd',
						loss='categorical_crossentropy',
						metrics=['accuracy'])
my_new_model.fit_generator(train_generator,
							steps_per_epoch=900,
							epochs=2,
							validation_data=validation_generator,
							validation_steps=400)


model_json = my_new_model.to_json()
with open("model_arch.json", "w") as json_file:
    json_file.write(model_json)
my_new_model.save_weights("my_model_weights.h5")