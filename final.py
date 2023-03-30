from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2
import numpy as np

with_mask=np.load("with_mask.npy")
without_mask=np.load("without_mask.npy")
print(with_mask.shape)
print(without_mask.shape)
with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)
print(with_mask.shape)
X=np.r_[with_mask,without_mask]
print(X.shape)
labels=np.zeros(X.shape[0])
labels[200:]=1.0
names={0:"Mask",1:"No Mask"}

x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.20)
# print(x_train.shape)

pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)

haar_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

capture=cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img=capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        # haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            face_fetch=img[y:y+h,x:x+w, :]
            face_fetch=cv2.resize(face_fetch,(50,50))
            face_fetch=face_fetch.reshape(1,-1)
            face_fetch=pca.transform(face_fetch)
            pred = svm.predict(face_fetch)[0]

            n = names[int(pred)]
            cv2.putText(img,n,(x-10,y-11),font,1,(0,0,255),2)
            # print(n)
        cv2.imshow("test", img)
        # 27 is ascii value of escape
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()