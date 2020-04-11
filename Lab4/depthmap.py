#Import libraries
from pylab import *
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

#SML
def sml(src,Ngd,deld = 50.50) :
    #Obtain partial double derivates (hessian)
    kfxx = array([[0,0,0],[1,-2,1],[0,0,0]])
    kfyy = array([[0,1,0],[0,-2,0],[0,1,0]])
    
    kext = len(kfxx)//2
    Nimg,Nr,Nc = shape(src)
    img = np.zeros((Nimg,int(Nr+2*kext),int(Nc+2*kext)))
    fin = np.zeros(shape(src))
    img[:,kext:Nr+kext,kext:Nc+kext] = np.copy(src)
    patch = np.zeros(shape(kfxx))
    #Go along rows and then along columns
    for i in range(kext,Nr+kext) :
        for j in range(kext,Nc+kext) :
            patch = img[:,i-kext:i+kext+1,j-kext:j+kext+1]
            patch1 = sum(patch*kfxx,axis = (1,2))
            patch2 = sum(patch*kfyy,axis = (1,2))
            #Modified by using abs
            patch = abs(patch1) + abs(patch2)
            fin[:,i-kext,j-kext] = patch
    if Ngd != 0 :        
        #sum the Ngd
        ksum = np.ones((int(2*Ngd+1),int(2*Ngd+1)))/(2*Ngd+1) 
        kext = len(ksum)//2
        Nimg,Nr,Nc = shape(fin)
        img = np.zeros((Nimg,int(Nr+2*kext),int(Nc+2*kext)))
        sml = np.zeros(shape(fin))
        img[:,kext:Nr+kext,kext:Nc+kext] = fin
        patch = np.zeros(shape(ksum))
        #Go along rows and then along columns
        for i in range(kext,Nr+kext) :
            for j in range(kext,Nc+kext) :
                patch = img[:,i-kext:i+kext+1,j-kext:j+kext+1]
                patch1 = sum(patch*ksum,axis = (1,2))
                sml[:,i-kext,j-kext] = patch1 
    else :
        sml = fin  
    depth = np.argmax(sml,axis = 0)  
    depth = depth*deld 
    return depth

#Loading .mat file
pics = loadmat("stack.mat")
#Loading only the images
imgs = []
for i in range(1,10) :
    img = pics["frame00%1d"%i]
    imgs.append(img)
    
for i in range(10,100) :
    img = pics["frame0%2d"%i]
    imgs.append(img)
           
img = pics["frame100"]
imgs.append(img)
imgs = array(imgs)
#print(shape(imgs))

#Test img plot
#plt.imshow(imgs[90])
#plt.show()
#Give input to sml
depth = sml(src = imgs,Ngd = 0)
#print(depth)

#Plot depth map
X = np.arange(len(depth))
Y = np.arange(len(depth[0]))
X,Y = meshgrid(X,Y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, depth, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)
#plt.legend()
plt.show()
