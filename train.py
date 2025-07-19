import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


INPUT_SIZE = (160, 160)
LR = 1e-4
MARGIN = 0.2
EPOCHS = 5
BATCH_SIZE = 32


def preprocess_face(face, size=INPUT_SIZE):
	face = cv2.resize(face, size)
	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = keras.applications.mobilenet_v2.preprocess_input(face.astype('float32'))
	return face


def build_classification_model(n_classes):
	base = keras.applications.MobileNetV2(include_top=False, input_shape=INPUT_SIZE + (3, ), pooling='avg')
	for layer in base.layers[:-30]:
		layer.trainable = False
	x = keras.layers.Dense(256, activation='relu')(base.output)
	x = keras.layers.Dropout(0.3)(x)
	output = keras.layers.Dense(n_classes, activation='softmax')(x)
	model = keras.Model(base.input, output)
	model.compile(optimizer=keras.optimizers.Adam(LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


def triplet_loss(_, y_pred):
	a, p, n = tf.split(y_pred, 3, axis=0)
	d_ap = tf.reduce_sum(tf.square(a - p), 1)
	d_an = tf.reduce_sum(tf.square(a - n), 1)
	return tf.reduce_mean(tf.maximum(d_ap - d_an + MARGIN, 0.0))


def build_metric_embedding():
	base = keras.applications.MobileNetV2(include_top=False, input_shape=INPUT_SIZE + (3, ), pooling='avg')
	x = keras.layers.Dense(256)(base.output)
	x = keras.layers.UnitNormalization(axis=1)(x)
	model = keras.Model(base.input, x)
	model.compile(optimizer=keras.optimizers.Adam(LR), loss=triplet_loss)
	return model


def make_triplet_dataset(root, batch_size):
	classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
	label2imgs = {c: [os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))] for c in classes}
	while True:
		A, P, N = [], [], []
		for _ in range(batch_size):
			a_lbl = random.choice(classes)
			n_lbl = random.choice([l for l in classes if l != a_lbl])
			a_img, p_img = random.sample(label2imgs[a_lbl], 2)
			n_img = random.choice(label2imgs[n_lbl])
			A.append(preprocess_face(cv2.imread(a_img)))
			P.append(preprocess_face(cv2.imread(p_img)))
			N.append(preprocess_face(cv2.imread(n_img)))
		yield np.concatenate([A, P, N], 0), np.zeros((batch_size * 3, ))


def train_classification(train_dir, val_dir, epochs, batch_size):
	# Training data with augmentation
	train_datagen = keras.preprocessing.image.ImageDataGenerator(
		preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
		rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
	)
	# Validation data without augmentation (only rescaling)
	val_datagen = keras.preprocessing.image.ImageDataGenerator(
		preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
	)
	
	train = train_datagen.flow_from_directory(
		train_dir, target_size=INPUT_SIZE, batch_size=batch_size, class_mode='sparse'
	)
	val = val_datagen.flow_from_directory(
		val_dir, target_size=INPUT_SIZE, batch_size=batch_size, class_mode='sparse'
	)
	
	model = build_classification_model(train.num_classes)
	callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
	model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)
	model.save('models/classification_model.keras')
	embedding_model = keras.Model(model.input, model.layers[-2].output)
	embedding_model.save('models/classification_embedding.keras')


def train_metric(train_dir, epochs, batch_size):
	model = build_metric_embedding()
	dataset = make_triplet_dataset(train_dir, batch_size)
	classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
	steps_per_epoch = max(100, len(classes) * 2)
	callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
	model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
	model.save('models/metric_embedding.keras')


if __name__ == '__main__':
	train_dir = 'dataset/classification_data/train_data'
	val_dir = 'dataset/classification_data/val_data'
	train_classification(train_dir, val_dir, EPOCHS, BATCH_SIZE)
	train_metric(train_dir, EPOCHS, BATCH_SIZE)