#Import libraries
from pylab import *
import cv2

#Load image
#path  = r"Globe.pgm"
path  = r"Nautilus.pgm"
src = cv2.imread(path,0)
N = shape(src)[0]

def varblur(src) :
    Nr,Nc = shape(src)
    fin = np.zeros(shape(src))
    #Go along rows and then along columns
    kmax = 13
    for i in range(kmax,Nr+kmax) :
        for j in range(kmax,Nc+kmax) :
            #Obtain blur kernel-sigma values
            A = 2
            B = (N*N/2)/(-np.log(0.01/A))  
            ioff = i-kmax
            joff = j-kmax
            sig = 1
            
            #Apply kernel on the image
            #print(sig)
            kernel = generate_kernel(sig)
            kext = len(kernel)//2
            
            img = np.zeros((int(Nr+2*kmax),int(Nc+2*kmax)))
            
            img[kmax:Nr+kmax,kmax:Nc+kmax] = src
            patch = np.zeros(shape(kernel))
            patch = img[i-kext:i+kext+1,j-kext:j+kext+1]
            patch = patch*kernel
            #print(sum(patch))
            fin[i-kmax,j-kmax] = sum(patch)
            #print(fin[i-kmax,j-kmax])
    #print(len(np.where(fin==0)[0]))        
    #print(fin)        
    return fin
    
def invblur(src,kernel) :
    kext = len(kernel)//2
    Nr,Nc = shape(src)
    img = np.zeros((int(Nr+2*kext),int(Nc+2*kext)))
    fin = np.zeros(shape(src))
    img[kext:Nr+kext,kext:Nc+kext] = src
    patch = np.zeros(shape(kernel))
    #Go along rows and then along columns
    for i in range(kext,Nr+kext) :
        for j in range(kext,Nc+kext) :
            patch = img[i-kext:i+kext+1,j-kext:j+kext+1]
            patch = patch*kernel
            fin[i-kext,j-kext] = sum(patch)
            
    return fin

#Obtain kernel for given sigma    
def generate_kernel(sig) :
    k = int(ceil(6*sig +1))
    if k%2 == 0 :
        k = k+1
    kernel = np.zeros((k,k))
    mid = k//2
    for i in range(0,mid+1) : 
        row = np.arange(mid+i,k)
        roweff = row-mid
        kernel[mid-i,row] = (1/(2*pi*sig*sig))*exp(-(roweff*roweff + i*i)/(2*sig*sig))
        kernel[mid-roweff[1:],mid+i] = kernel[mid-i,row][1:]
    
    kernel[:mid+1,:mid] = np.fliplr(kernel[:mid+1,mid+1:])  
    kernel[mid+1:,:] = np.flipud(kernel[:mid,:])
    kernel = kernel/sum(kernel)
    return kernel 
    
sig = 1    
tgt = varblur(src)
kernel = generate_kernel(sig)
#print(kernel)
tgt2 = invblur(src,kernel)

#Plots
plt.imshow(tgt, cmap = "gray")
plt.axis("off")
plt.show()

plt.imshow(tgt2, cmap = "gray")
plt.axis("off")
plt.show()
plt.imshow(abs(tgt-tgt2),cmap = "gray")
plt.axis("off")
plt.show()
