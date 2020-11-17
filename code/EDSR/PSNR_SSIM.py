import numpy as np
import math,os
import cv2
from PIL import Image
from scipy.signal import convolve2d

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)

    ref_data = np.array(ref,dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps

    if(rmse == 0):
        rmse = eps
    return 20*math.log10(255.0/rmse)


def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)
    # 通道分离，注意顺序BGR不是RGB

    (B1, G1, R1) = cv2.split(imageA)

    (B2, G2, R2) = cv2.split(imageB)

    # convert the images to grayscale BGR2GRAY

    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 方法一

    # (grayScore, diff) = compute_ssim(grayA, grayB)
    #
    # diff = (diff * 255).astype("uint8")

    #print("gray SSIM: {}".format(grayScore))

    # 方法二

    score0 = compute_ssim(B1, B2)

    score1 = compute_ssim(G1, G2)

    score2 = compute_ssim(R1, R2)

    aveScore = (score0+score1+score2)/3

    #print("BGR average SSIM: {}".format(aveScore ))

    return aveScore

top_path = r"G:\ye\Postgraduate\MyCode\Python-Project\SR\Cell_SR\PPT\data\3"
img_name = "61_overlay512.tif"
low_img_name = "61_overlay256.tif"
pred_img_name = "61_overlay256_pred.png"
nature_img_name = "Composite-1.tif"
target = cv2.imread(os.path.join(top_path,img_name))
low = cv2.imread(os.path.join(top_path,low_img_name))
pred_img = cv2.imread(os.path.join(top_path,pred_img_name))
nature_img = cv2.imread(os.path.join(top_path,nature_img_name))
bicubic = cv2.resize(low,(512,512),interpolation=cv2.INTER_LINEAR)
print("-----------bicubic-----------")
print(psnr(target,bicubic))
print(ssim(target,bicubic))
print("-----------pred-----------")
print(psnr(target,pred_img))
print(ssim(target,pred_img))
print("-----------nature-----------")
print(psnr(target,nature_img))
print(ssim(target,nature_img))
