#Import libraries
from pylab import *
import cv2
import random as rn
import csv
from scipy.linalg import null_space

#Functions 

#Bilinear
def bilnr(src, H, xy) :
     #Mapping to src grid
	xy_temp = np.linalg.inv(H) @ xy
	x = xy_temp[0]/xy_temp[2]
	y = xy_temp[1]/xy_temp[2]
	
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

#RANSAC
def ransac(corresp1,corresp2) :
     frac = 0
     niter = 0
     while(frac <= 0.98) :
             
          #Generate 4 random numbers from the set
          Lmp = len(corresp1)
          r = rn.sample(range(0,Lmp),4)
          a = [corresp1[r[i]] for i in range(0,len(r))]
          b = [corresp2[r[i]] for i in range(0,len(r))] 
          #Take these 4 points and find homography
          #Fill in the matrix
          vsize = 9
          eqns = 4
          A = np.zeros((int(2*eqns),vsize))
          #print(shape(A))
          #Loop to fill in the values
          for i in range(0,eqns) :
               A[int(2*i)][0] = b[i][0]
               A[int(2*i)][1] = b[i][1]
               A[int(2*i)][2] = 1
               A[int(2*i)][3] = 0
               A[int(2*i)][4] = 0
               A[int(2*i)][5] = 0
               A[int(2*i)][6] = -b[i][0]*a[i][0]
               A[int(2*i)][7] = -b[i][1]*a[i][0]
               A[int(2*i)][8] = -a[i][0]
               
               A[int(2*i)+1][0] = 0
               A[int(2*i)+1][1] = 0
               A[int(2*i)+1][2] = 0
               A[int(2*i)+1][3] = b[i][0]
               A[int(2*i)+1][4] = b[i][1]
               A[int(2*i)+1][5] = 1
               A[int(2*i)+1][6] = -b[i][0]*a[i][1]
               A[int(2*i)+1][7] = -b[i][1]*a[i][1]
               A[int(2*i)+1][8] = -a[i][1]

          #Find nullspace of the matrix
          h = null_space(A)
          #print(shape(h))
          #Put h in order
          H = h.reshape((3,3))

          #Check with remaining points and see fraction
          C = []
          iterset = list(set(np.arange(0,Lmp)).difference(r))
          bvec = np.zeros((3,1))
          avec = np.zeros((2,1))
          eps = 10
          bvec[2] = 1
          for item in iterset :
               bvec[0] = corresp2[item][0]
               bvec[1] = corresp2[item][1]
               
               atemp = H @ bvec
               avec[0] = atemp[0]/atemp[2]
               avec[1] = atemp[1]/atemp[2]
               
               dist = np.sqrt(pow(corresp1[item][0]-avec[0],2) + pow(corresp1[item][1]-avec[1],2))
               if dist < eps :
                    C.append(item)
               
          #Check how good in the consensus set
          frac = len(C)/len(iterset)
          niter = niter+1
     return H,frac,niter,C      


#Run SIFT and obtain matching key points
corresp1 = []
corresp2 = []
with open("corresp.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        corresp1.append(row[:2])
        corresp2.append(row[2:])
corresp1 = array(corresp1)
corresp2 = array(corresp2)

H,frac,niter,C = ransac(corresp1,corresp2)
print(frac)
print(H)
print(niter)

#Define path of src
path2  = r"/home/vignesh/EE5175/Lab2/IMG1.pgm"
src2 = cv2.imread(path2,0)

path1  = r"/home/vignesh/EE5175/Lab2/IMG2.pgm"
src1 = cv2.imread(path1,0)

#Initialize tgt
tgt = np.zeros(shape(src2))

for xn in range(0,len(tgt)) :
     for yn in range(0,len(tgt[0])) :
          
          xy = array([xn,yn,1])
          #print("xn=",xn)
          tgt[xn][yn] = bilnr(src1, H, xy)
          
#print(tgt)
#THE TRANSFOMRATION IS ROTATION OF 30 DEGREES AND TRANSLATION 5.48 AND 155.63 (X AND Y RESPECTIVELY)
plt.imshow(tgt,cmap = "gray")
plt.show()
plt.imshow(tgt-src2,cmap = "gray")
plt.show()
#print(tgt)
