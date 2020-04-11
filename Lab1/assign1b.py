#Import libraries
from pylab import *
import cv2

#Bilinear function
def bilnr(src,theta, xn, yn) :
     #Mapping to src grid
	x = np.cos(theta)*xn - np.sin(theta)*yn
	y = np.sin(theta)*xn + np.cos(theta)*yn
	xf = int(np.floor(x)) +cenx
	yf = int(np.floor(y)) +ceny
	
	#distance from pixel
	a = x+cenx-xf
	b = y+ceny-yf
	
	#print(xf)
	
	#Calculate intensity
	if xf >= shape(src)[0]-1 or yf >=shape(src)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	     #Ival = src[xf][yf]
	return Ival

#Define path of src
path  = r"/home/vignesh/EE5175/Assignments/Lab1/pisa_rotate.pgm"
src = cv2.imread(path,0)
#print(shape(src))
#Define centre
cenx = int(floor(shape(src)[0]/2))
ceny = int(floor(shape(src)[1]/2))
#Translation
theta = -4*pi/180

#Initialize tgt
tgt = np.zeros(shape(src))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
     
          #print("xn=",xn)
          tgt[xn][yn] = bilnr(src, theta, xn-cenx, yn-ceny)
          
#print(tgt)

#FINAL OUTPUT HAS A ROTATION OF 4 DEGREES ABOUT THE CENTRE COMPARED TO THE ORIGINAL IMAGE
plt.imshow(tgt,cmap = "gray")
plt.show()
#print(tgt)
