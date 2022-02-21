import numpy as np

#np.array()
vec=np.array([1,2,3,4,5])
print("vec: ", vec)

mat=np.array([[10,20,30], [60,70,80]])
print("mat: ", mat)

print('vec의 타입: ', type(vec))
print('mat의 타입: ', type(mat))

print("vec의 축의 개수: ", vec.ndim)
print("vec의 크기(shape): ", vec.shape)

print("mat의 축의 개수: ", mat.ndim)
print("mat의 크기(shape): ", mat.shape, end='\n\n\n')

#ndarray의 초기화
zero_mat=np.zeros((2,3))
print('zero_mat: ', zero_mat)

one_mat=np.ones((2,3))
print('one_mat: ', one_mat)

same_value_mat=np.full((2,2), 7)
print("full (2,2) to 7: ", same_value_mat)

eye_mat=np.eye(3)
print("eye_mat(3): ", eye_mat)

random_mat=np.random.random((2,2))
print('random_mat: ', random_mat, end='\n\n\n')

#np.arange()
range_vec=np.arange(10)
print('range_vec(10): ', range_vec)

n=2
range_n_step_vec=np.arange(1, 10, n)#has step
print('range_n_step_vec(2): ', range_n_step_vec, end='\n\n\n')

#np.reshape()
reshape_mat=np.array(np.arange(30)).reshape((5,6))
print("reshaped(5,6) map : ", reshape_mat, end='\n\n\n')

#Numpy Slicing
mat=np.array([[1,2,3], [4,5,6]])
print("mat: ", mat)

slicing_mat=mat[0,:]#2차원 0이고 1차원 전부
print("slicing_mat[0,:]: ", slicing_mat)

slicing_mat=mat[:, 1]#2차원 전부고 1차원 index1
print("slicing_mat[:,1]: ", slicing_mat, end='\n\n\n')

#Numpy integer indexing
mat=np.array([[1,2], [4,5], [7,8]])
print("mat: ", mat)

print("mat[1,0]:", mat[1,0])#point special location

indexing_mat=mat[[2,1],[1,1]]#Make new array by slicing. by two element in that place
print("mat[[2,1],[1,1]]: ", indexing_mat, end='\n\n\n')

#Numpy 연산_element-wise
x=np.array([1,2,3])
y=np.array([4,5,6])
print("x: ", x, "y: ", y)

result=x+y
print("x+y: ", result)

result=x-y
print("x-y: ", result)

result=result*x
print("result*x: ", result)

result=result/x
print("result/x: ", result,end='\n\n')

mat1=np.array([[1,2], [3,4]])
mat2=np.array([[5,6], [7,8]])
mat3=np.dot(mat1, mat2)#Multiplication of Matrix by dot()
print("mat1: ", mat1, "\nmat2: ", mat2, "\ndot(mat1, mat2): ", mat3)
