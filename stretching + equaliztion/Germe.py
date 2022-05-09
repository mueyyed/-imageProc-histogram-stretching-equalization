from cv2 import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

#histogram  
def Hist(image) :
    H = np.zeros((256,1))
    s = image.shape 
    for i in range( s[0]) :
        for j in range(s[1]) : 
            k= image[i, j] 
            H[k, 0 ] = H[k, 0 ] + 1 
    return H 

Goruntu = cv2.imread( "test2.png") 
Goruntu = cv2.resize( Goruntu , ( 600 , 400)) 
s = Goruntu.shape 
GoruntuGray = cv2.cvtColor( Goruntu , cv2.COLOR_BGR2GRAY) 
# manipulasyon Konstrast ve Parlaklik 
GoruntuGray = cv2.convertScaleAbs( GoruntuGray , alpha = 1.10 , beta = 40 )

cv2.imshow( 'Orginal Goruntu ' , GoruntuGray ) 

histg = Hist( GoruntuGray) 
plt.plot(histg) 
plt.show()


Genilik = histg.reshape( 1 , 256) 
Uzunluk = np.zeros((1,256))
 
for i in range( 256):
     if Genilik[0,i] == 0:
         Uzunluk[0,i]=0
     else:
        Uzunluk[0,i]=i
	
min = np.min(Uzunluk[np.nonzero(Uzunluk)]) 
max = np.max( Uzunluk[np.nonzero(Uzunluk)]) 
	
Germe = np.round(((255-0) / ( max -min))* ( Uzunluk-min)) 
Germe[ Germe < 0 ] =0 
Germe[Germe> 255] = 255 
	
for i in range ( s[0]) : 
	for j in range( s[1]): 
            k = GoruntuGray[i,j]
            GoruntuGray[i,j] = Germe[0,k]
            
histg2 = Hist(GoruntuGray) 
cv2.imshow( 'GermeGoruntu' , GoruntuGray) 
plt.plot(histg) 
plt.plot(histg2) 
plt.show()


cv2.waitKey(0) 