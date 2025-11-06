import cv2
import scipy.signal as signal
import numpy as np

def smoth_image(np_list,x_v=0,y_v=0):
    edit_matrix=np_list#copy.copy(np_list)
    for num in range(x_v):
        temp_matrix=edit_matrix
        for raw_id in range(edit_matrix.shape[0]-1):
            new_raw=edit_matrix[[raw_id,raw_id+1]].mean(axis=0)
            temp_matrix=np.insert(temp_matrix,int(2*raw_id + 1),new_raw,axis=0)
        edit_matrix=temp_matrix
    
    
    for num in range(y_v):
        temp_matrix=edit_matrix
        for raw_id in range(edit_matrix.shape[1]-1):
            new_raw=edit_matrix[:,[raw_id,raw_id+1]].mean(axis=1)
            temp_matrix=np.insert(temp_matrix,int(2*raw_id + 1),new_raw,axis=1)
        edit_matrix=temp_matrix
    return edit_matrix 
    
def gaussBlur(image,sigma=5,H=3,W=3,_boundary = 'fill', _fillvalue = 0):
    #水平方向上的高斯卷积核
    gaussKenrnel_x = cv2.getGaussianKernel(sigma,W,cv2.CV_64F)
    #进行转置
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    #图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image,gaussKenrnel_x,mode='same',boundary=_boundary,fillvalue=_fillvalue)
    #构建垂直方向上的卷积核
    gaussKenrnel_y = cv2.getGaussianKernel(sigma,H,cv2.CV_64F)
    #图像与垂直方向上的高斯核卷积核
    gaussBlur_xy = signal.convolve2d(gaussBlur_x,gaussKenrnel_y,mode='same',boundary= _boundary,fillvalue=_fillvalue)
    return gaussBlur_xy
    
if __name__=="__main__":
    t1=np.array( [ [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4] ] )
    t1_s=smoth_image(t1,2,2)
    t1_sf=(t1_s)
    print(t1_sf)