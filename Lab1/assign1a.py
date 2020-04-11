#Import libraries
from pylab import *
import cv2

#Bilinear function
def bilnr(src,tx, ty, xn, yn) :
     #Mapping to src grid
	x = xn-(ty-1)
	y = yn-(tx-1)
	xf = int(np.floor(x))
	yf = int(np.floor(y))
	
	#distance from pixel
	a = x-xf
	b = y-yf
	
	#Calculate intensity
	if xf >= shape(src)[0]-1 or yf >=shape(src)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	
	return Ival

#Define path of src
path  = r"lena_translate.pgm"
src = cv2.imread(path,0)
print(shape(src))
#cv2.imshow("Source",src)
#cv2.waitKey(0)

#Translation
tx = 3.75
ty = 4.3

#Initialize tgt
tgt = np.zeros(shape(src))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          tgt[xn][yn] = bilnr(src, tx, ty, xn, yn)
          
#print(tgt[:6])
#cv2.imwrite("image",tgt)
#cv2.imread("image
plt.imshow(tgt,cmap = "gray")
plt.show()
#print(tgt)
