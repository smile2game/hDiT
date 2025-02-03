import numpy as np
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])

# 目标数组的形状是 (4, 5) 
# (5,6) >> (1,1) = 1*5+1 = 6
indices = np.ravel_multi_index((5, 6), (4, 5), mode='wrap')
print(indices)
