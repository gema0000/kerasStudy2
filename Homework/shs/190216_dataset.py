import numpy as np

x1 = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([101,102,103,104,105,106,107,108,109,110])

y1 = np.array([1,2,3,4,5,6,7,8,9,10])
y2 = np.array([101,102,103,104,105,106,107,108,109,110])

# x3 = np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])
# y3 = np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])

x3 = np.array(range(1001,1101))
y3 = np.array(range(1001,1101))

print('x1.shape :', x1.shape) 
print('type(x1) :', type(x1))
print(x1)

x1_train = x1[:7]
x2_train = x2[:7]   
x3_train = x3[:70]
x3_train = x3_train.reshape(10,7)
merXtra = np.vstack((x1_train, x2_train, x3_train))
merXtra = np.transpose(merXtra)
print("merXtra = ", merXtra)

y1_train = y1[:7]
y2_train = y2[:7]
y3_train = y3[:70]
y3_train = y3_train.reshape(10,7)
merYtra = np.vstack((y1_train, y2_train, y3_train))
merYtra = np.transpose(merYtra)
print("merYtra = ", merYtra)

x1_test = x1[7:]  
x2_test = x2[7:]  
x3_test = x3[70:]
x3_test = x3_test.reshape(10,3)
merXtest = np.vstack((x1_test, x2_test, x3_test))
merXtest = np.transpose(merXtest)
print("merXtest = ", merXtest)

y1_test =y1[7:]  
y2_test =y2[7:]
y3_test =y3[70:]
y3_test = y3_test.reshape(10,3)
merYtest = np.vstack((y1_test, y2_test, y3_test))
merYtest = np.transpose(merYtest)
print("merYtest = ", merYtest)