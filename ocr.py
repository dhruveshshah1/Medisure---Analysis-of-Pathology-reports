import cv2
import pytesseract
import re
import tensorflow as tf
import numpy as np
import pandas as pd



from tensorflow import keras



img = cv2.imread("x.PNG")
custom_config = r'-c tessedit_char_blacklist=:/()*;|$\'\-‘£- --oem 3 --psm 6'
x = pytesseract.image_to_string(img, config=custom_config)

pattern4 = re.compile(r'PATIENTr\'s VALUES\s+([0-9.\s])+Unit')
p=[]
p.append(re.compile(r'(?s)HAEMOGLOBIN(.*?)g'))
p.append(re.compile(r'(?s)RBC COUNT(.*?)M'))
p.append(re.compile(r'(?s)P.C(.*?)MALE'))
p.append(re.compile(r'(?s)M.C.V.(.*?)Cubic'))
p.append(re.compile(r'(?s)M.C.H.(.*?)pico'))
p.append(re.compile(r'(?s)M.C.H.C.(.*?)M'))
p.append(re.compile(r'(?s)PLATELETS(.*?)LACS'))
# p.append(re.compile(r'(?s)TOTAL WBC COUNT(.*?)CuMM'))
print(x)
g=[]
for i in p:
    g.append(i.finditer(x))

values = []

for r in g:
    for i in r:
        values.append(i.group())

#WORD MATCH HOTFIX
for i in range(len(values)):
    values[i]=values[i].replace("%","")


values[1]=values[1].replace('RBC COUNT','')


# REVERSING MATCHED STRINGS
rev_values=[]
for i in values:
    rev_values.append(i[::-1])

#EXTRACTING DIGITS
d = re.compile(r'(?s) (.*?) ')
d2 = re.compile(r'(?s) +(.*?) ')
f=d.finditer(rev_values[0])
extracted_digits_iterator=[]
for i in rev_values:
    extracted_digits_iterator.append(d2.finditer(i))

cleaned_values_reversed=[]
for i in extracted_digits_iterator:
    for j in i:
        cleaned_values_reversed.append(j.group())



print("==================================")
# print(cleaned_values_reversed)

cleaned_values=[]
for i in cleaned_values_reversed:
    cleaned_values.append(i[::-1])



# REMOVE WHITESPACES

for i in range(len(cleaned_values)):
    cleaned_values[i]=cleaned_values[i].rstrip()
    cleaned_values[i]=cleaned_values[i].lstrip()
    cleaned_values[i]=cleaned_values[i].replace(",","")

# print(cleaned_values)

#REMOVES COMMA FOR WBC
# o=cleaned_values[-1].replace(",","")

#CASTING TO FLOAT
float_values=[]
for i in cleaned_values:
    float_values.append(float(i))
float_values[-1]=float_values[-1]*100



#SUCCESS


# print(float_values)
bloodwork_values=float_values
print("=================BLOOD WORK VALUES=================")
print(bloodwork_values)


#DIFFRENTIAL COUNT
dfcount_reg=[]
dfcount_reg.append(re.compile(r'(?s)NEUTROPHILS(.*?)%'))
dfcount_reg.append(re.compile(r'(?s)LYMPHOCYTES(.*?)%'))
dfcount_reg.append(re.compile(r'(?s)MONOCYTES(.*?)%'))
dfcount_reg.append(re.compile(r'(?s)BASOPHILS(.*?)%'))
dfcount_reg.append(re.compile(r'(?s)FORMS(.*?)%'))

dfcount_iterator=[]

for i in dfcount_reg:
    dfcount_iterator.append(i.finditer(x))

dfcount_uncleaned_values=[]

for r in dfcount_iterator:
    for i in r:
        dfcount_uncleaned_values.append(i.group())

# print(dfcount_uncleaned_values)

#STATIC CLEANING

dfcount_uncleaned_values[0]=dfcount_uncleaned_values[0].replace("NEUTROPHILS ","")
dfcount_uncleaned_values[0]=dfcount_uncleaned_values[0].replace(" %","")

dfcount_uncleaned_values[1]=dfcount_uncleaned_values[1].replace("LYMPHOCYTES ","")
dfcount_uncleaned_values[1]=dfcount_uncleaned_values[1].replace(" %","")

dfcount_uncleaned_values[2]=dfcount_uncleaned_values[2].replace("MONOCYTES ","")
dfcount_uncleaned_values[2]=dfcount_uncleaned_values[2].replace(" %","")

dfcount_uncleaned_values[3]=dfcount_uncleaned_values[3].replace("BASOPHILS ","")
dfcount_uncleaned_values[3]=dfcount_uncleaned_values[3].replace(" %","")

dfcount_uncleaned_values[4]=dfcount_uncleaned_values[4].replace("FORMS ","")
dfcount_uncleaned_values[4]=dfcount_uncleaned_values[4].replace(" %","")

dfcount_cleaned_values = []

for i in dfcount_uncleaned_values:
    dfcount_cleaned_values.append(float(i))

# print(dfcount_cleaned_values)

# SUCCESS

dfcount_values=dfcount_cleaned_values

print("=================DIFFRENTIAL COUNT VALUES=================")
print(dfcount_values)



#RBC MORPHOLOGY


morpho_reg=[]
morpho_reg.append(re.compile(r'(?s)MICROCYTES(.*?)ABSENT|PRESENT'))
morpho_reg.append(re.compile(r'(?s)MACROCYTES(.*?)ABSENT|PRESENT'))
morpho_reg.append(re.compile(r'(?s)STIPPLING(.*?)ABSENT|PRESENT'))
morpho_reg.append(re.compile(r'(?s)TARGET CELLS(.*?)ABSENT|PRESENT'))
morpho_reg.append(re.compile(r'(?s)TEAR DROP CELL(.*?)ABSENT|PRESENT'))

morpho_iterator=[]

for i in morpho_reg:
    morpho_iterator.append(i.finditer(x))

morpho_uncleaned=[]

for i in morpho_iterator:
    for j in i:
        morpho_uncleaned.append(j.group())


#REVERSE IT <<<<<<<<<<<<<<<<<<<<<<<<<
morpho_uncleaned_reversed=[]
for i in morpho_uncleaned:
    morpho_uncleaned_reversed.append(i[::-1])

#SLICE IT----------------
morpho_uncleaned_reversed_sliced=[]
for i in morpho_uncleaned_reversed:
    morpho_uncleaned_reversed_sliced.append(i[:6])


#REVERSE AGAIN :O       CAN't ToUCH THIs :O
morpho_cleaned_txt=[]
for i in morpho_uncleaned_reversed_sliced:
    i=i.rstrip(" ")
    morpho_cleaned_txt.append(i[::-1])


# print(morpho_cleaned_txt)

morpho_values=[]
for i in morpho_cleaned_txt:
    if i=="ABSENT":
        morpho_values.append(float("0"))
    else:
        morpho_values.append(float("1"))


print("=================RBC . MORPHOLOGY VALUES=================")
print(morpho_values)

gender=[]
gender.append(float("0"))
dummy=[]
dummy.append(float("0"))
empty_bullets= gender + bloodwork_values + dfcount_values + morpho_values
print(empty_bullets)

shells = np.asarray(empty_bullets, dtype = float)

# print(type(live_bullets))

#LOADING TRAINING DATA

data = pd.read_csv('train6k.csv')
x_train=data.iloc[:,0:18].values
y_train=data.iloc[:,19:].values


#LOADING MODELS
guns = tf.keras.models.load_model('prototype_model')
loss, acc = guns.evaluate(x_train, y_train, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

live_bullets = shells.reshape((1,18))

target=guns.predict(live_bullets)
print("=====================THE PREDICTION======================")


chart=["IRON Deficiency", "B12 Deficiency", "Lukemia", "Anemia", "Polycythemia Vera", "Genetic Anemia","Infection/Inflammation","Malaria","Dengue","Aids","Thalassaemia","Bone Marrow","Jaundice","Blood  Tranfusion", "NONE"]
for i in range(len(target[0])):
    print("{} : {:.2f}%".format(chart[i] ,target[0][i] * 100))
