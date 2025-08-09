import cv2
import mss
import numpy as np
import pytesseract
import pyautogui
import ssl
import urllib.request
import urllib
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from pytesseract import image_to_string
# headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.6) Gecko/20040206 Firefox/0.8","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3","Accept-Encoding": "none","Accept-Language": "en-US,en;q=0.8","Connection": "keep-alive",}
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
# request = urllib.request.Request("https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe", None, headers=headers)
# response = urllib.request.urlopen(request, context=ctx)
# url_file = response.read()
# with open("./tesseract-ocr-w64-setup-5.5.0.20241111.exe", "wb") as fi:
#     fi.write(url_file)
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
_imr = lambda x: (int(abs(x[2]*(x[0]/x[1]))),int(abs(x[3])))if x[0]<x[1]else(int(abs(x[2])),int(abs(x[2]//(x[0]/x[1]))))if x[0]>x[1]else(x[2],x[3])
_HW_KERN_ = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
_BLB=lambda blb:cv2.bilateralFilter(blb[0],blb[1],blb[2],blb[3])
_F2D=lambda f2d:cv2.filter2D(f2d[0], f2d[1], f2d[2])
_GAU=lambda gau:cv2.GaussianBlur(gau[0],(gau[1],gau[2]),gau[3])
RGB_PALETTE = {"bw":np.array([[0,0,0],[255,255,255]]),"3c":np.array([[0,0,0],[128,128,128],[255,255,255]]),"7c":np.array([[0,0,0],[36,36,36],[72,72,72],[109,109,109],[145,145,145],[182,182,182],[218,218,218],[255,255,255]]),'36g':np.array([[0,0,0],[7,7,7],[14,14,14],[21,21,21],[29,29,29],[36,36,36],[43,43,43],[51,51,51],[58,58,58],[65,65,65],[72,72,72],[80,80,80],[87,87,87],[94,94,94],[102,102,102],[109,109,109],[116,116,116],[123,123,123],[131,131,131],[138,138,138],[145,145,145],[153,153,153],[160,160,160],[167,167,167],[174,174,174],[182,182,182],[189,189,189],[196,196,196],[204,204,204],[211,211,211],[218,218,218],[225,225,225],[233,233,233],[240,240,240],[247,247,247],[255,255,255]]),}
_CKD=lambda cp,img:np.uint8(RGB_PALETTE[cp][cKDTree(RGB_PALETTE[cp]).query(img[:,:,::-1],k=1)[1]])[:,:,::-1]
# with mss.mss() as sct:
#     monitor = sct.monitors[2]
#     sct_img = sct.grab(monitor)
#     img:np.uint8 = cv2.cvtColor(np.uint8(sct_img), cv2.COLOR_RGB2BGR)
img = cv2.imread("Y:/captcha/202508041345.png",cv2.IMREAD_ANYCOLOR)
img = np.uint8(img)
imax = np.max([img.shape[0]*4,img.shape[1]*4])
_rsz = (img.shape[1],img.shape[0],int(imax//2),int(imax//2))
img =  cv2.bitwise_not(cv2.resize(_CKD('bw',_CKD('3c',_CKD('7c',_CKD('36g',img)))),_imr(_rsz)))
img = _BLB([img,9,75,75])
img = _GAU((img,7,7,0))
img = _F2D((img,-1,_HW_KERN_))
print(img.shape)
raw_str:str = image_to_string(
    img,lang='eng',config='--psm 11 --oem 1 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyz')
raw_str = raw_str[raw_str.find('chat')+4:].strip().split(chr(10))[0]
print(raw_str)
plt.imshow(img)
# pyautogui.press('t')
# for i,x in enumerate(raw_str):
#     pyautogui.press(str(x))
