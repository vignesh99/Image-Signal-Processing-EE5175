#Import libraries
from pylab import *
import cv2

#Load image
path  = r"Mandrill.pgm"
src = cv2.imread(path,0)
sigma = array([1.6, 1.2, 1, 0.6, 0.3, 0])
#sigma = array([1.6])

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
    

for i in range(0,len(sigma)) :
    sig = sigma[i]
    if sig == 0 :
        plt.imshow(src, cmap = "gray")
        plt.show()
    else :  
        kernel = generate_kernel(sig)
        #print(kernel)
        tgt = invblur(src,kernel)
        #Plots
        plt.imshow(tgt, cmap = "gray")
        plt.axis("off")
        plt.show()

#print(shape(src))


