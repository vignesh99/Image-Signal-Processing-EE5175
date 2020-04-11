#Import libraries
from pylab import *
import cv2

#Bilinear function
def bilnr(src,R, T, xy) :
     #Mapping to src grid
	xy_temp = R @ (xy-T) #- array([7,27])
	x = xy_temp[0]
	y = xy_temp[1]
	
	xf = int(np.floor(x)) 
	yf = int(np.floor(y)) 
	
	#distance from pixel
	a = x-xf
	b = y-yf
	
	#print(xf)
	
	#Calculate intensity
	if xf >= shape(src)[0]-1 or yf >=shape(src)[1]-1 or xf<=0 or yf<= 0 :
	     Ival = 0
	else :
	     Ival = (1-a)*(1-b)*src[xf][yf] + (1-a)*(b)*src[xf][yf+1] + (a)*(1-b)*src[xf+1][yf] + (a)*(b)*src[xf+1][yf+1]
	     #Ival = src[xf][yf]
	return Ival

#Define path of src
path1  = r"/home/vignesh/EE5175/Assignments/Lab1/IMG1.pgm"
src1 = cv2.imread(path1,0)

path2  = r"/home/vignesh/EE5175/Assignments/Lab1/IMG2.pgm"
src2 = cv2.imread(path2,0)

#print(shape(src1)) (296,512)
#print(shape(src2)) (517,598)

#Image points
i1a = array([125,30])
i2a = array([249,94])
i1b = array([373,158])
i2b = array([400,329])

#Solving for the parameters
tx = i2a[0]-i2b[0]
ty = i2a[1]-i2b[1]
sx = i1a[0]-i1b[0]
sy = i1a[1]-i1b[1]

t = np.zeros((2,1))
s = np.zeros((2,2))
t[0] = tx
t[1] = ty

s[0][0] = sx
s[0][1] = sy
s[1][0] = sy
s[1][1] = -sx

ab = np.linalg.inv(s) @ t
R = np.zeros((2,2))
R[0][0] = ab[0]
R[0][1] = ab[1]
R[1][0] = -ab[1]
R[1][1] = ab[0]

T = i2a - (R @ i1a)
T = T[::-1]
print(T,R)
#(80,112) ; 30deg

#Initialize tgt
tgt = np.zeros(shape(src2))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          
          xy = array([xn,yn])
          #print("xn=",xn)
          tgt[xn][yn] = bilnr(src1, R, T, xy)
          
#print(tgt)
#THE TRANSFOMRATION IS ROTATION OF 30 DEGREES AND TRANSLATION 5.48 AND 155.63 (X AND Y RESPECTIVELY)
plt.imshow(tgt,cmap = "gray")
plt.show()
plt.imshow(src2-tgt,cmap = "gray")
plt.show()
#print(tgt)
