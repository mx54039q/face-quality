#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# DCT: sharpness
def get_dct(img):
    dct = cv2.dct(img)
    s = img.shape
    idct = np.zeros(s,dtype='float32')
    idct[:s[0]*3/4,:s[1]*3/4] = dct[:s[0]*3/4,:s[1]*3/4]
    idct = cv2.idct(idct)
    return np.square(idct-img).mean()

# Histogram: symmetry
def his_sym(img):
    his1 = cv2.calcHist(img[:,:30], [0], None, [16], [0.,255.])
    his2 = cv2.calcHist(img[:,30:], [0], None, [16], [0.,255.])
    return np.square(his1 - his2).mean()

# Histogram: contrst
def his_con(img):
    his = cv2.calcHist(img, [0], None, [16], [0.,255.])
    return np.square(his - his.mean()).mean()

# Sobel: sharpness
def get_sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    z = cv2.Sobel(img, cv2.CV_16S, 1, 1)
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)
    absZ = cv2.convertScaleAbs(z) 
    dst1 = cv2.addWeighted(absX,0.5,absY,0.5,0)
    dst = cv2.addWeighted(absZ,0.5,dst1,0.5,0)
    return dst.mean()

# Nuclear norm
def nu_norm(img):
    return np.linalg.norm(img,ord='nuc')    

imglist = np.loadtxt('probe.txt',dtype='string',delimiter=' ')
imglist = pd.DataFrame(imglist,columns=['path','ori_pose','ori_ill'])
gallery = imglist[(imglist.ori_pose=='051') & (imglist.ori_ill=='07')]
probe = imglist[(imglist.ori_ill!='07') & (imglist.ori_pose!='051')]
sobel_ccbr,dct_ccbr,norm_ccbr,con_ccbr = [],[],[],[]
sobel_cpf,dct_cpf,norm_cpf,con_cpf = [],[],[],[]
for i in range(probe.shape[0]):
    print i
    name = probe.iloc[i,0][:-3]
    img_ccbr = np.float32(cv2.resize(cv2.imread('ccbr_probe/'+name+'jpg',0)[15:165,15:135],(50,54)))
    img_cpf = np.float32(cv2.imread('cpf_probe/'+name+'bmp',0)[1:55,5:55])
    sobel_ccbr.append(get_sobel(img_ccbr));sobel_cpf.append(get_sobel(img_cpf))
    dct_ccbr.append(get_dct(img_ccbr));dct_cpf.append(get_dct(img_cpf))
    norm_ccbr.append(nu_norm(img_ccbr));norm_cpf.append(nu_norm(img_cpf))
    con_ccbr.append(his_con(img_ccbr));con_cpf.append(his_con(img_cpf))
    #plt.imshow(img,cmap ='gray');plt.show()

probe_ccbr = probe.copy()
probe_ccbr['sobel']= sobel_ccbr
probe_ccbr['dct']= dct_ccbr
probe_ccbr['norm']= norm_ccbr
probe_ccbr['con']= con_ccbr
probe_ccbr.to_csv('quality_ccbr.csv',index=False)

probe_cpf = probe.copy()
probe_cpf['sobel']= sobel_cpf
probe_cpf['dct']= dct_cpf
probe_cpf['norm']= norm_cpf
probe_cpf['con']= con_cpf
probe_cpf.to_csv('quality_cpf.csv',index=False)    
    
    
    
    
    
