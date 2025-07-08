import numpy as np      
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from metabci.brainda.algorithms.decomposition import SKLDA
from metabci.brainda.algorithms.decomposition import STDA

Xtrain = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])     #The training set
y = np.array([1, 1, 1, 2, 2, 2])                                              #lables of the training set
Xtest = np.array([[-0.8, -1], [-1.2, -1], [1.2, 1], [0.5, 2]])                #The test set

#LDA demo
#By learning an optimal hyperplane, the original high-dimensional data is projected to the low-dimensional
# plane through the hyperplane, and the data samples are classified in the new plane.
clf = LinearDiscriminantAnalysis()      
clf.fit(Xtrain, y)           #Training the LDA classification model
print('LDA分类正确率为：',np.sum(np.where(clf.predict(Xtest)-np.array([1,1,2,2]),0,1))/4*100,'%')   #LDA预测值

#SKLDA demo
#Shrinkage Linear Discriminant Analysis (SKLDA) can reduce the data dimension by optimizing local features.
#SKLDA can improve the small-sample problem of LDA algorithm to some extent.
clf2 = SKLDA()
clf2.fit(Xtrain, y)          #Train SKLDA classification model
#print(clf2.transform(Xtest)) #SKLDA decision value
print('SKLDA分类正确率为：',np.sum(np.where(np.sign(clf2.transform(Xtest))-np.array([-1,-1,1,1]),0,1))/4*100,'%') 

#STDA demo
#The STDA algorithm learns two projection matrices by alternating collaborative optimization of the spatial and
#temporal dimensions of EEG, so as to maximize the discriminability of the projected features between the target
#class and the non-target class.
#The two projection matrices are used to transform the constructed spatial-tmporal two-dimensional samples into
#new one-dimensional samples with significantly reduced dimensionality, which effectively improves the estimation
#of covariance matrix parameters and enhances the generalization ability of the classifier under small training sample sets.
Xtrain2 = np.random.randint(-10, 10, (100*2, 16, 19))                        #The training set
y2 = np.hstack((np.ones(100, dtype=int), np.ones(100, dtype=int) * -1))      #lables of the training set
Xtest2 = np.random.randint(0, 10, (4, 16, 19))                              #The test set

clf3 = STDA()
clf3.fit(Xtrain2, y2)
z=clf3.transform(Xtest2)      #Train STDA classification model
#print(clf3.transform(Xtest2)) #STDA decision value
print('STDA分类正确率为：',np.sum(np.where(np.sign(clf3.transform(Xtest2))-np.array([1,1,1,1]),0,1))/4*100,'%') 


