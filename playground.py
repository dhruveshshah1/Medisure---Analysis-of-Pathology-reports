from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
import cv2
import pytesseract
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import random


empty_bullets=[1.0,14.0,4.93,33.0,66.90,28.30 ,42.40 ,2.36, 59, 36.0, 1.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0]

shells = np.asarray(empty_bullets, dtype = float)

data = pd.read_csv('train6k.csv')
x_train=data.iloc[:,0:18].values
y_train=data.iloc[:,19:].values


# guns = tf.keras.models.load_model('prototype_model')
# loss, acc = guns.evaluate(x_train, y_train, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
#
# live_bullets = shells.reshape((1,18))
#
# target=guns.predict(live_bullets)
# print("THE PREDICTIONNNN")


# chart=["IRON Deficiency", "B12 Deficiency", "Lukemia", "Anemia", "Polycythemia Vera", "Genetic Anemia","Infection/Inflammation","Malaria","Dengue","Aids","Thalassaemia","Bone Marrow","Jaundice","Blood  Tranfusion"]
# for i in range(len(target[0])):
#     print("{} : {:.2f}%".format(chart[i] ,target[0][i] * 100))

model = tf.keras.models.load_model('prototype_model')
loss, acc = model.evaluate(x_train, y_train, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))




_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(x_train)

chart=["IRON Deficiency", "B12 Deficiency", "Lukemia", "Anemia", "Polycythemia Vera", "Genetic Anemia","Infection/Inflammation","Malaria","Dengue","Aids","Thalassaemia","Bone Marrow","Jaundice","Blood  Tranfusion", "NONE"]
random_check = random.randint(0, 3498)
random_check2 = random_check
for i in range(random_check,random_check+1):
	print("====================RANDOM TEST===========")
	print("Entry No:", random_check)
	for j in range(len(predictions[0])):
		print("{} : {:.2f}%".format(chart[j], predictions[i][j] * 100))
	print("EXPECTED :", y_train[i])
	print("-------------------")