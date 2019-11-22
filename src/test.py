import numpy as np

print('\\')
k=np.load('../dataset/train/avg.npy')
print(k.shape)
#input()
for i in k:
    print(i)