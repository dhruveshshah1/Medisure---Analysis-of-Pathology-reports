import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape

from livelossplot import PlotLossesKeras





data = pd.read_csv('train6k.csv')

x=data.iloc[:,0:18].values

y=data.iloc[:,19:].values

data1 = pd.read_csv('testing6k.csv')

x_train=data1.iloc[:,0:18].values

y_train=data1.iloc[:,19:].values



opn=15
nc=2
# model = Sequential()
# model.add(Dense(24, input_dim=18, activation='relu'))
# model.add(Dense(20, activation='relu'))
# # model.add(Dense(14, activation='softmax'))
# model.add(Dense(15, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(x, y, epochs=500, batch_size=100,callbacks=[PlotLossesKeras()],validation_data=(x_train, y_train))


model = Sequential()
model.add(Dense(24, input_dim=18, activation='relu'))
model.add(Dense(20, activation='relu'))
# model.add(Dense(14, activation='softmax'))
model.add(Dense(15, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x, y, epochs=100, batch_size=100,validation_data=(x_train, y_train))


# model.save('prototype_model123')

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

# predictions = model.predict(x)
#
# print(x)
# chart=["IRON Deficiency", "B12 Deficiency", "Lukemia", "Anemia", "Polycythemia Vera", "Genetic Anemia","Infection/Inflammation","Malaria","Dengue","Aids","Thalassaemia","Bone Marrow","Jaundice","Blood  Tranfusion", "NONE"]
# for i in range(7):
# 	for j in range(len(predictions[0])):
# 		print("{} : {:.2f}%".format(chart[j], predictions[i][j] * 100))
# 	print("EXPECTED :", y[i])
# 	print("-------------------")
