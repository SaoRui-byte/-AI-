import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import time
from sklearn.svm import SVC

print('1.加载数据')
minst = datasets.load_digits()
x,y = minst.data,minst.target
# print(x)
# print(y)

print('2.划分数据')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)
# print(x_test.shape)
# print(x_train.shape)

print('3.训练模型')
start = time.time()
Svm = SVC(C=1.0,max_iter=300,random_state=42,kernel='rbf',gamma='scale',verbose=True)
Svm.fit(x_train,y_train)
end = time.time()
print('训练时间：',end-start)

print('4.模型预测')
y_predict = Svm.predict(x_test)
print('准确率：',round(accuracy_score(y_test,y_predict),3))
