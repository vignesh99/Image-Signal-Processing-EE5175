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
	b = y  -yf
	
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
     while(frac <= 0.95) :
             
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

#Define path of src
path1  = r"/home/vignesh/EE5175/Assignments/Lab2/mosaic1.pgm"
src1 = cv2.imread(path1,0)

path2  = r"/home/vignesh/EE5175/Assignments/Lab2/mosaic2.pgm"
src2 = cv2.imread(path2,0)

path3  = r"/home/vignesh/EE5175/Assignments/Lab2/mosaic3.pgm"
src3 = cv2.imread(path3,0)

#Run SIFT and obtain matching key points
correspa1 = []
correspa2 = []
with open("corresp_mosaic2_1.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        correspa1.append(row[:2])
        correspa2.append(row[2:])
correspa1 = array(correspa1)
correspa2 = array(correspa2)

#Run SIFT and obtain matching key points
correspc1 = []
correspc2 = []
with open("corresp_mosaic2_3.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        correspc1.append(row[:2])
        correspc2.append(row[2:])
correspc1 = array(correspc1)
correspc2 = array(correspc2)

H1,frac1,niter1,C1 = ransac(correspa1,correspa2)
H3,frac3,niter3,C3 = ransac(correspc1,correspc2)
print(frac1,frac3)
print(H1,H3)
print(niter1,niter3)

'''
#Initialize tgt
tgt1 = np.zeros(shape(src2))

for xn in range(0,len(tgt1)) :
     for yn in range(0,len(tgt1[0])) :
          
          xy = array([xn,yn,1])
          #print("xn=",xn)
          tgt1[xn][yn] = bilnr(src1, H1, xy)
#plt.imshow(tgt1,cmap = "gray")
#plt.show()
          
tgt3 = np.zeros(shape(src2))

for xn in range(0,len(tgt3)) :
     for yn in range(0,len(tgt3[0])) :
          
          xy = array([xn,yn,1])
          #print("xn=",xn)
          tgt3[xn][yn] = bilnr(src3, H3, xy)
          
#print(tgt)
#THE TRANSFOMRATION IS ROTATION OF 30 DEGREES AND TRANSLATION 5.48 AND 155.63 (X AND Y RESPECTIVELY)
plt.imshow(src2-tgt1,cmap = "gray")
plt.show()
plt.imshow(src2-tgt3,cmap = "gray")
plt.show()

'''
#Define topcorer
cenx = int(floor(shape(src1)[1]))

#Create canvas
nrows = shape(src2)[0]
ncolumns = shape(src1)[1] + shape(src2)[1] + shape(src3)[1]
print(shape(src1)[1],shape(src2)[1])

canvas = np.zeros((nrows,ncolumns))
canvas1 = np.zeros((nrows,ncolumns))
canvas2 = np.zeros((nrows,ncolumns))
canvas3 = np.zeros((nrows,ncolumns))
countcnvs = np.ones((nrows,ncolumns))
#countcnvs3 = np.zeros((nrows,ncolumns))
'''
for xn in range(0,len(canvas)) :
     for yn in range(0,len(canvas[0])) :
          
          xy = array([xn,yn-cenx,1])
          #print("xn=",xn)
          canvas[xn][yn] = bilnr(src2, np.identity(len(xy)), xy)

plt.imshow(canvas,cmap = "gray")
plt.show()
'''
for xn in range(0,len(canvas)) :
     for yn in range(0,len(canvas[0])) :
          
          xy = array([xn,yn-cenx,1])
          #print("xn=",xn)
          canvas1[xn][yn] = bilnr(src1, H1, xy)
          if canvas1[xn][yn] != 0 :
               countcnvs[xn][yn] = countcnvs[xn][yn]+ 1

for xn in range(0,len(canvas)) :
     for yn in range(0,len(canvas[0])) :
          
          xy = array([xn,yn-cenx,1])
          #print("xn=",xn)
          canvas3[xn][yn] = bilnr(src3, H3, xy)
          if canvas3[xn][yn] != 0 :
               countcnvs[xn][yn] = countcnvs[xn][yn] + 1
               
for xn in range(0,len(canvas)) :
     for yn in range(0,len(canvas[0])) :
          
          xy = array([xn,yn-cenx,1])
          #print("xn=",xn)
          canvas2[xn][yn] = bilnr(src2, np.identity(len(xy)), xy)
          if canvas2[xn][yn] == 0 and countcnvs[xn][yn] == 3 :
               countcnvs[xn][yn] = 2
          if canvas2[xn][yn] == 0 and countcnvs[xn][yn] == 2 :
               countcnvs[xn][yn] = 1
 
canvas = canvas1 + canvas3 + canvas2 
plt.imshow(canvas/countcnvs,cmap = "gray")
plt.show()
