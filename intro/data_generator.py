import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# location of your dataset:
train_dir = './'
validation_dir = './'

# training data generator:
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(300, 300),
	batch_size=128,
	class_mode='binary')

# training data generator:
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(300, 300),
	batch_size=32,
	class_mode='binary'
)

# model definition:
model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(16, (3, 3),
                           activation='relu',
                           input_shape=(300, 300, 3)),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])
# print(model.summary())

# model compiling:
model.compile(optimizer=RMSprop(lr=0.001),
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

# model fitting:
model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2
)