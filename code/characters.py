import numpy as np
import pylab as pl

def draw_bar(start, end, img_shape, thickness):

    start = np.asarray(start).ravel()
    end = np.asarray(end).ravel()
    
    grid=np.mgrid[0.:1.:1. / img_shape[0], 0.:1.:1. / img_shape[1]]
    grid=grid.reshape(2, -1)
    R = np.ones((2,2))
    R[:, 0] = end - start
    norm = np.sqrt(R[1][0]*R[1][0]+R[0][0]*R[0][0])
    R[0][1]=R[1][0]/norm
    R[1][1]=-R[0][0]/norm
    new_grid=np.dot(np.linalg.inv(R), np.subtract(grid,start[:, np.newaxis]))
    img=np.zeros(img_shape)
    img=((new_grid[0]>=0.)*(new_grid[0]<=1.)*(np.abs(new_grid[1])<=thickness/2)).reshape(img_shape).astype(np.float64)
    return img

# def signatures_to_images(M, img_shape, thickness): 
# result = np.zeros(img_shape)
# N=M.reshape(4,-1)
# for m in N:
# result=hstack(result,np.zeros(img_shape))
# if(m[0]):
# result=result+draw_bar((0,0),(img_shape[0]/2,0),img_shape, thickness) 
# if(m[1]): 
# result=result+draw_bar((img_shape[0]/2,0),(img_shape[0],0),img_shape, thickness) 
# if(m[2]): 
# result=result+draw_bar((0,img_shape[1]/2),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# if(m[3]): 
# result=result+draw_bar((img_shape[0]/2,img_shape[1]/2),(img_shape[0],img_shape[1]/2),img_shape, thickness) 
# if(m[4]): 
# result=result+draw_bar((0,img_shape[1]),(img_shape[0]/2,img_shape[1]),img_shape, thickness) 
# if(m[5]): 
# result=result+draw_bar((img_shape[0]/2,img_shape[1]),(img_shape[0],img_shape[1]),img_shape, thickness) 
# if(m[6]): 
# result=result+draw_bar((0,0),(0,img_shape[1]/2),img_shape, thickness) 
# if(m[7]): 
# result=result+draw_bar((0,img_shape[1]/2),(0,img_shape[1]),img_shape, thickness) 
# if(m[8]): 
# result=result+draw_bar((img_shape[0]/2,0),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# if(m[9]): 
# result=result+draw_bar((img_shape[0]/2,img_shape[1]/2),(img_shape[0]/2,img_shape[1]),img_shape, thickness) 
# if(m[10]): 
# result=result+draw_bar((img_shape[0],0),(img_shape[0],img_shape[1]/2),img_shape, thickness) 
# if(m[11]): 
# result=result+draw_bar((img_shape[0],img_shape[1]/2),(img_shape[0],img_shape[1]),img_shape, thickness) 
# if(m[12]): 
# result=result+draw_bar((0,0),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# if(m[13]): 
# result=result+draw_bar((0,img_shape[1]),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# if(m[14]): 
# result=result+draw_bar((img_shape[0],0),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# if(m[15]): 
# result=result+draw_bar((img_shape[0],img_shape[1]),(img_shape[0]/2,img_shape[1]/2),img_shape, thickness) 
# return result

# def char_to_signatures(S): 
#   n=len(S) 
# 	M=np.zeros((n,16)) 
# 	for i in np.arange(n): 
# 		if(S[i]=='A'): 
# 			M[i]=[1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0] 
# 		elif(S[i]=='B'): 
# 			M[i]=[0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,0] 
# 		elif(S[i]=='C'): 
# 			M[i]=[1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='D'): 
# 			M[i]=[0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='E'): 
# 			M[i]=[1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0] 
# 		elif(S[i]=='F'): 
# 			M[i]=[1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0] 
# 		elif(S[i]=='G'): 
# 			M[i]=[1,1,0,0,0,1,1,1,0,1,1,1,0,0,0,0] 
# 		elif(S[i]=='H'): 
# 			M[i]=[1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0] 
# 		elif(S[i]=='I'): 
# 			M[i]=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='J'): 
# 			M[i]=[0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,0] 
# 		elif(S[i]=='K'): 
# 			M[i]=[0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,1] 
# 		elif(S[i]=='L'): 
# 			M[i]=[1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='M'): 
# 			M[i]=[1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0] 
# 		elif(S[i]=='N'): 
# 			M[i]=[1,1,0,0,1,1,0,0,0,0,0,0,1,0,0,1] 
# 		elif(S[i]=='O'): 
# 			M[i]=[1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='P'): 
# 			M[i]=[1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0] 
# 		elif(S[i]=='Q'): 
# 			M[i]=[1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,1] 
# 		elif(S[i]=='R'): 
# 			M[i]=[1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,1] 
# 		elif(S[i]=='S'): 
# 			M[i]=[1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0] 
# 		elif(S[i]=='T'): 
# 			M[i]=[0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0] 
# 		elif(S[i]=='U'): 
# 			M[i]=[1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0] 
# 		elif(S[i]=='V'): 
# 			M[i]=[1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0] 
# 		elif(S[i]=='W'): 
# 			M[i]=[1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1] 
# 		elif(S[i]=='X'): 
# 			M[i]=[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1] 
# 		elif(S[i]=='Y'): 
# 			M[i]=[0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0] 
# 		elif(S[i]=='Z'): 
# 			M[i]=[0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,0] 
# 		else: 
# 			print"Probleme" 
# 	return M

if __name__ == "__main__":
    b = draw_bar([.9, .5], [.25, .75], (100, 100), .1)

    pl.figure()
    pl.imshow(b, interpolation="nearest")
    pl.gray()
    pl.show()
