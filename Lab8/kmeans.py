#Import libraries
from pylab import *
import cv2
from random import *
set_printoptions(threshold=sys.maxsize)

#Load image
K = 3               #No of clusters
N = 30              #No of random intializations
path1  = r"flower.png"
src1 = cv2.imread(path1)
path2  = r"car.ppm"
src2 = cv2.imread(path2)

#Parameters
c1 = array([0,255,0])
c2 = array([0,0,0])
c3 = array([255,255,255])
C = array([c1,c2,c3])
#c4 = array([0,0,255])
#C = array([c1,c2,c3,c4])

def kmeans(img,K,C,Iter=5) :
                                    #Divide the dataset into K-clusters
    imgK = np.zeros((K,shape(img)[0],shape(img)[1]))
    newC = []
    ssq = 0
    indLbl = []
    for k in range(0,K) :           #Norm with respect to centres
        imgK[k] = norm(img-C[k],axis=2)
    label = argmin(imgK,axis=0)     #Fix the labels
    #print("0",len(np.where(label==0)[0]))
    #print("1",len(np.where(label==1)[0]))
    #print("2",len(np.where(label==2)[0]))
    
    #Recompute the pattern centres
    for k in range(0,K) :
        ind = np.where(label==k)      
        indLbl.append(ind)
        clstr = img[ind]            #Compute Sum of squares
        ssq = ssq + sum(norm(clstr-C[k],axis=1)**2)                       
        newC.append(clstr.mean(axis=0).astype(int))
    
    #print(newC)
    #print(len(indLbl))
    newC = array(newC)
    if array_equal(newC,C) :
        return newC,indLbl,ssq
    #Recusrion and check if label are same
    Iter = Iter-1
    if Iter <= 0 :
        return newC,indLbl,ssq
    cntr,indx,ssq = kmeans(img,K,newC,Iter)
    return cntr,indx,ssq
    
#plot the images
C,indLbl,ssq = kmeans(src1,K,C)
img1 = np.zeros(shape(src1))
img1[indLbl[0]] = [0,255,0]
img1[indLbl[1]] = [255,0,0]
img1[indLbl[2]] = [0,0,255]
plt.imshow(img1)
plt.show()
C,indLbl,ssq = kmeans(src2,K,C)
img2 = np.zeros(shape(src2))
img2[indLbl[0]] = [0,255,0]
img2[indLbl[1]] = [255,0,0]
img2[indLbl[2]] = [0,0,255]
plt.imshow(img2)
plt.show()

#Random initializations
#All centres and ssq
sqsum1 = []
sqsum2 = []
centres = []

for n in range(0,N) :
    #Generate random pixels
    c1 = sample(range(0,256),3)
    c2 = sample(range(0,256),3)
    c3 = sample(range(0,256),3)
    C = array([c1,c2,c3])
    centres.append(C)
    C,indLbl,ssq = kmeans(src1,K,C,1)
    sqsum1.append(ssq)
    C,indLbl,ssq = kmeans(src2,K,C,1)
    sqsum2.append(ssq)
    
print("MAX IMG1",centres[sqsum1.index(max(sqsum1))])
print("MIN IMG1",centres[sqsum1.index(min(sqsum1))])
print("MAX IMG2",centres[sqsum2.index(max(sqsum2))])
print("MIN IMG2",centres[sqsum2.index(min(sqsum2))])
    
#plot the images
C,indLbl,ssq = kmeans(src1,K,centres[sqsum1.index(min(sqsum1))])
img1 = np.zeros(shape(src1))
img1[indLbl[0]] = [0,255,0]
img1[indLbl[1]] = [255,0,0]
img1[indLbl[2]] = [0,0,255]
plt.imshow(img1)
plt.show()
C,indLbl,ssq = kmeans(src2,K,centres[sqsum2.index(min(sqsum2))])
img2 = np.zeros(shape(src2))
img2[indLbl[0]] = [0,255,0]
img2[indLbl[1]] = [255,0,0]
img2[indLbl[2]] = [0,0,255]
plt.imshow(img2)
plt.show()
