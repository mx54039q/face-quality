import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from skimage.feature import local_binary_pattern as lbp
import math
import pandas as pd

def load_lra_train(iset, number, resize = 1):  
    if(iset == 'ccbr'):
        image_shape1 = [number*140,188,153]
        itype = 'jpg'
    else:
        image_shape1 = [number*140,60,60]
        itype = 'bmp'
    trainData = np.zeros((image_shape1))
    probe = np.loadtxt('probe.txt',dtype='string',delimiter=' ')
    for i in range(number*140):  
        iname = iset+'_probe/'+probe[i,0][:-3]+itype
        trainData[i,:,:] = np.float32(cv2.imread(iname,0))
    for i in range(number):
        trainData[i*140:(i+1)*140,:,:] -= trainData[i*140:(i+1)*140,:,:].mean(axis=0)
    return trainData

def load_gallery(iset, resize = 1):
    if(iset == 'ccbr'):
        image_shape1 = [int(188/resize),int(153/resize)]
        itype = 'jpg'
    else:
        image_shape1 = [int(60/resize),int(60/resize)]
        itype = 'bmp'
    trainData = np.zeros(([149]+image_shape1))    
    yTrain=range(149)
    gallery = np.loadtxt('gallery.txt',dtype='string',delimiter=' ')
    h,w = image_shape1[1],image_shape1[0]
    for i in range(149):
        iname = iset+'_gallery/'+gallery[i,0][3:-3]+itype
        trainData[i,:,:] = np.float32(cv2.resize(cv2.imread(iname,0),(h,w)))
    return trainData, np.array(yTrain)
#
def calHistogram(ImgLBPope, h_size = 4, w_size = 4):
    Img = ImgLBPope
    H,W = Img.shape
    Img = Img.astype('float32')
    Histogram = np.zeros((h_size*w_size,128))
    maskx,masky = W/w_size,H/h_size
    for i in range(h_size):
        for j in range(w_size):       
            mask = Img[i*masky: (i+1)*masky,j*maskx :(j+1)*maskx]
            hist = cv2.calcHist([mask],[0],None,[128],[0.,255.])
            Histogram[i*w_size+j,:] = hist.flatten()
    return Histogram.flatten()

def cos_dis(v1,v2):
    """
    cosine distance
    """
    return 1 - np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# Load Image
def loadImage(iset, resize = 1):  
    if(iset == 'ccbr'):
        image_shape1 = [149,188,153];image_shape2 = [16986,188,153]
        itype = 'jpg'
    else:
        image_shape1 = [149,60,60];image_shape2 = [16986,60,60]
        itype = 'bmp'
    trainData = np.zeros((image_shape1)); testData = np.zeros((image_shape2))
    yTrain=range(149); yTest = np.repeat(range(149)*19,6);
    gallery = np.loadtxt('gallery.txt',dtype='string',delimiter=' ')
    probe = np.loadtxt('probe.txt',dtype='string',delimiter=' ')
    for i in range(149):
        iname = iset+'_gallery/'+gallery[i,0][3:-3]+itype
        trainData[i,:,:] = np.float32(cv2.imread(iname,0))
    for i in range(16986):  
        iname = iset+'_probe/'+probe[i,0][:-3]+itype
        testData[i,:,:] = np.float32(cv2.imread(iname,0))
    return trainData, testData, np.array(yTrain), np.array(yTest)

######################### HOG Classifier
def HOG_classifier():
    gallery = np.loadtxt('gallery.txt',dtype='string',delimiter=' ')
    probe = np.loadtxt('probe.txt',dtype='string',delimiter=' ')
    probe = pd.DataFrame(probe,columns=['path','ori_pose','ori_ill'])
    probe_pose = probe[probe.ori_pose=='050'].values
    label = range(149)*19
    feat_gallery = np.zeros((149,1312200))
    winSize = (128,128);blockSize = (16,16);blockStride = (8,8);cellSize = (8,8)
    nbins = 9;derivAperture = 1;winSigma = 4.;histogramNormType = 0;
    L2HysThreshold = 2.0000000000000001e-01;gammaCorrection = 0;nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
        derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    for i in range(149):
        if(iset == 'ccbr'):name = 'ccbr_gallery/'+gallery[i,0][3:-3]+'jpg'
        else:name = 'cpf_gallery/'+gallery[i,0][3:-3]+'bmp'
        #img = cv2.resize(cv2.imread(name,0),(200,200))
        img = cv2.imread(name,0)
        feat_gallery[i,:] = hog.compute(img,(4,4),(4,4)).flatten()
    
    W = np.dot(np.eye(149),np.linalg.pinv(feat_gallery.T))
    pose = ['041', '050', '080', '130','140', '190']
    label = range(149)*19
    total = []
    for ipose in pose:
        count = 0
        probe_pose = probe[probe.ori_pose==ipose].values
        for i in range(probe_pose.shape[0]):
            if(iset == 'ccbr'):name = 'ccbr_probe/'+probe_pose[i,0][:-3]+'jpg'
            else:name = 'cpf_probe/'+probe_pose[i,0][:-3]+'bmp'
            #img = cv2.resize(cv2.imread(name,0),(200,200))
            img = cv2.imread(name,0)
            feat = hog.compute(img,(4,4),(4,4)).flatten()
            ilabel = np.dot(W,feat).argmax()
            if(ilabel == label[i]):
                count = count+1
            print('Current Acc: %d / %d , %f' % (count,i,count/float(i+1)))
        print('Total Acc: %d / %d , %f' % (count,i,count/float(i+1)))
        total.append(count/float(i+1))
