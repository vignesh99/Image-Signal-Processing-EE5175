#Import libraries
from pylab import *
import cv2

#Bilinear function
def bilnr(src, sclx, scly, xn, yn) :
     #Mapping to src grid
	x = xn/sclx
	y = yn/scly
	xf = int(np.floor(x)) +cenx
	yf = int(np.floor(y)) +ceny
	
	#distance from pixel
	a = x+cenx-xf
	b = y+ceny-yf
	
	#Calculate intensity
	if xf >= shape(src)[0]-1 or yf >=shape(src)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	
	return Ival

#Define path of src
path  = r"/home/vignesh/EE5175/Assignments/Lab1/cells_scale.pgm"
src = cv2.imread(path,0)
#print(shape(src)[0])
#Define centre
cenx = int(floor(shape(src)[0]/2))
ceny = int(floor(shape(src)[1]/2))
#Translation
sclx = 0.8
scly = 0.8
#Initialize tgt
tgt = np.zeros(shape(src))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          tgt[xn][yn] = bilnr(src, sclx, scly, xn-cenx, yn-ceny)
          
print(tgt[50])
plt.imshow(tgt,cmap = "gray")
plt.show()
#print(tgt)
