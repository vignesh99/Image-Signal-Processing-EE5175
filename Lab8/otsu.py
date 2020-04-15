#Import libraries
from pylab import *
import cv2

#Load image
path1  = r"palmleaf1.pgm"
src1 = cv2.imread(path1,0)
path2  = r"palmleaf2.pgm"
src2 = cv2.imread(path2,0)

#Find global threshold
def gthres(F,L) :
    nL = np.arange(0,L)
    N = sum(F)
    muT = (sum(nL*F))/N
    sigwin = []
    siginb = []
    for t in range(0,L) :
        #Find parameters
        N1 = sum(F[:t+1])
        N2 = sum(F[t+1:])
        mu1 = (sum(nL[:t+1]*F[:t+1]))/N1
        mu2 = (sum(nL[t+1:]*F[t+1:]))/N2
        sig1 = (sum(((nL[:t+1]-mu1)**2)*F[:t+1]))/N1
        sig2 = (sum(((nL[t+1:]-mu2)**2)*F[t+1:]))/N2
        sigval1 = (sig1*N1 + sig2*N2)/N
        sigwin.append(sigval1)
        sigval2 = ((mu1-muT)*(mu1-muT)*N1 + (mu2-muT)*(mu2-muT)*N2)/N
        siginb.append(sigval2)
    #Find min thres
    gthres = np.where(sigwin == min(sigwin))[0]
    #gthres = np.where(siginb == max(siginb))[0]
    
    #Plot the sigma values
    plt.bar(nL,sigwin)
    plt.title("Minima plot of within class sigma")
    plt.xlabel("Threshold")
    plt.ylabel("Within class sigma")
    plt.show()
    
    return gthres
       
#Obtain intensity frequency
L = 256
F1 = []
F2 = []
for i in range(0,L) :
    ind1 = np.where(src1 == i)[0]
    ind2 = np.where(src2 == i)[0]
    F1.append(len(ind1))
    F2.append(len(ind2))
    
F1 = array(F1)
F2 = array(F2)

#Frequency plots    
plt.bar(range(0,L),F1)
plt.title("Frequency plot palmleaf1")
plt.xlabel("Threshold")
plt.ylabel("Frequency")
plt.show()
plt.bar(range(0,L),F2)
plt.title("Frequency plot palmleaf2")
plt.xlabel("Threshold")
plt.ylabel("Frequency")
plt.show()

#Thresholded pictures
t1 = gthres(F1,L)
t2 = gthres(F2,L)
print(t1,t2)

thresimg1 = np.zeros(shape(src1))
thresimg2 = np.zeros(shape(src2))

ind1 = np.where(src1 <= t1)
ind2 = np.where(src1 > t1)
thresimg1[ind1] = 0
thresimg1[ind2] = 1

ind1 = np.where(src2 <= t2)
ind2 = np.where(src2 > t2)
thresimg2[ind1] = 0
thresimg2[ind2] = 1

plt.imshow(thresimg1,cmap = "gray")
plt.show()
plt.imshow(thresimg2,cmap = "gray")
plt.show()

