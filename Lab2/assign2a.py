#Import libraries
from pylab import *
import cv2
import random as rn
import csv
from scipy.linalg import null_space

#Functions 

#Bilinear
#Rows and columns of the target as input
def bilnr(src, H, rows, cols) :
     #Creating vector to multiply Hinv
     x = list(np.arange(0,rows))
     x = x*cols
     x = array(x)
     x = x.reshape(cols,rows)
     x = x.T
     x = x.reshape(int(rows*cols),1)
     y = list(np.arange(0,cols))
     y = y*rows
     y = array(y)
     y = y.reshape(int(rows*cols),1)
     o = np.ones((int(rows*cols),1))
     xy = array([x,y,o])
     xy = xy.T
     xy = xy[0]
     xy = xy.T
     xy[1] = xy[1] - cenx
     
     #Target to source mapping  
     xy_temp = np.linalg.inv(H)@ xy
     #xy_temp = xy_temp.T
     x = xy_temp[0]/xy_temp[2]
     y = xy_temp[1]/xy_temp[2]
     
     #print(shape(x),shape(y))
     
     #xf = int(np.floor(x)) 
     #yf = int(np.floor(y))
     xf = x.astype(int)
     yf = y.astype(int)
	
	#distance from pixel
     a = x-xf
     b = y-yf
     
     Ival = np.zeros(shape(xf))
     #print(shape(src))
     #Find intensity
     for i in range(0,len(xf)) :
          #if check[i] == False :
          if xf[i] < shape(src)[0]-1 and yf[i] < shape(src)[1]-1 and xf[i]>0 and yf[i]>0 :
               #print(yf[i])
               Ival[i] = (1-a[i])*(1-b[i])*src[xf[i]][yf[i]] + (1-a[i])*(b[i])*src[xf[i]][yf[i]+1] + (a[i])*(1-b[i])*src[xf[i]+1][yf[i]] + (a[i])*(b[i])*src[xf[i]+1][yf[i]+1]

     Ival = Ival.reshape(rows,cols)
     return Ival

#RANSAC
def ransac(corresp1,corresp2) :
     frac = 0
     niter = 0
     while(frac <= 0.99) :
             
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
          #12 0.75 good
          #16 0.75 works
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

path1  = r"/home/vignesh/EE5175/Lab2/mosaic1.pgm"
src1 = cv2.imread(path1,0)

path2  = r"/home/vignesh/EE5175/Lab2/mosaic2.pgm"
src2 = cv2.imread(path2,0)

path3  = r"/home/vignesh/EE5175/Lab2/mosaic3.pgm"
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
#print(H1,H3)
#print(niter1,niter3)

#Define topcorer
#print(shape(src1))
cenx = int(floor(shape(src1)[1]))

#Create canvas
nrows = shape(src2)[0]
ncolumns = shape(src1)[1] + shape(src2)[1] + shape(src3)[1]
print(shape(src1)[1],shape(src2)[1])

canvas = np.zeros((nrows,ncolumns))
countcnvs = np.zeros((nrows,ncolumns))

canvas1 = bilnr(src1, H1, nrows, ncolumns)
canvas2 = bilnr(src2, np.identity(3), nrows, ncolumns)
canvas3 = bilnr(src3, H3, nrows, ncolumns)

#Finding no of intensities at each point
temp = np.equal(canvas1,np.zeros(shape(canvas1)))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

temp = np.equal(canvas2,np.zeros(shape(canvas1)))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

temp = np.equal(canvas3,np.zeros(shape(canvas1)))
temp = ~temp
temp = temp.astype(int)
countcnvs = countcnvs + temp

#Image plot
canvas = canvas1 + canvas3 + canvas2 
plt.imshow(canvas/countcnvs,cmap = "gray")
plt.axis("off")
plt.show()

