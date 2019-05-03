import numpy as np
import matplotlib.pylab as plt
from random import random

    
#first part generates a sample adjacency matrix (a) with k communities
#first part:
k=5 # case with k=5
n=1000
p=np.zeros((k, k)) #p is a kxk symmetric  matrix with elements in range [0,1], each element equals to the link probabilty between communities (off-diagonal) or 
#inside communities (diagonal), elements of p are uniformly random reals in range [0,1]

for i in range(k): #this for loop selects the random elements of p matrix
	for j in range(i,k):
	    p[i,j]=np.random.rand()
	    p[j,i]=p[i,j]
np.savetxt('p.txt', p, fmt='%.3f') #saves p-matrix
a=np.zeros((n,n))#a is the adjacency matrix
label=np.random.randint(k, size=n)# nodes have uniformly random labels from 0 to k-1, indicating community membership in one of k communities
for i in range(n):
	for j in range(i+1, n):
		ss=np.random.rand()
		if p[label[i],label[j]]<ss:#links are drawn randomly and independetly according to labeling of nodes iusing p-matrix
			a[i,j]=1
			a[j,i]=1
np.savetxt('a.csv',a,delimiter=',',fmt='%10.1f')	#saves a	

np.savetxt('label.txt', label, fmt='%i') #saves labels
b=np.add(-a,1) # b is auxliary matrix 'anti-adjasency' matrix 1-a
#second part: find the communities
tmax=10000 # number of optimization rounds, should be as large as possible
t=1
costglobal=10**36 # initial, large value of the cost function to be minimized
rglobal=np.zeros((n,k)) #initial value of best clustering matrix, nxk binary matrix, where rows indicate community membership of the corresponding node with value 1
while t < tmax:
  r=np.zeros((n,k)) # running clustering matrix, similar to rglobal
  for i in range(n):
  	m=np.random.randint(k)-1 # r starts from an uniformly random clustering
  	r[i,m]=1
  toi=1
  toimax=20 # maximal number of optimization cycles per t, should be large enough
  cost=10**30 # initial value of local cost function 
  while toi <toimax:
  	pp=np.dot(np.transpose(r),np.dot(a,r)) #kxk matrix 
  	nn=np.zeros(k) # list nn describe current sizes, number of nodes in each,  of communities
  	for i in range(k):
  		nn[i]=sum(np.transpose(r)[i])
  	if min(nn)<5: #small communities are not accepted, can be adjusted according to data size
  		toi=toimax+1
  	else:
  		toi+=1
  		e=pp
  		pe=pp
  		for i in range(k):
  			e[i,i]=pe[i,i]/2# is kxk matrix where elements are number of links between and within current communties
  		for i in range(k):
  			for j in range(i+1,k):
  				pe[i,j]=e[i,j]/nn[i]/nn[j] # pe is kxk matrix of link densitie between abd within current communties corresponding to r matrix
  				pe[j,i]=pe[i,j]
  		for i in range(k):
  			pe[i,i]=2*e[i,i]/nn[i]/(nn[i]-1)
  		log=np.log(np.add(pe,10**-34))
  		log2=np.log(np.add(np.add(-pe,1),10**-34))

  		for i in range(n):
  			b[i,i]=0
  		L=-np.dot(a,np.dot(r,log))-np.dot(b,np.dot(r,log2)) # L is a nxk cost matrix, each row shows k cost values of placing a node in various communities of the current r
  		currentcost=0
  		ff=[0 for i in range(n)]
  		for i in range(n):
  			f=np.argmin(L[i]) # finds optimal community for node i
  			ff[i]=int(f)
  			currentcost=currentcost+L[i,f] # computes the value of current cost function
  		rm=np.zeros((n,k))
  		for i in range(n):
  			rm[i,ff[i]]=1 # computes new clustering matrix rm
  		if(currentcost<cost):# new clustering matrix is accepted as the new current clustering if the total cost of clustering decreases
  			r=rm
  			cost=currentcost# value of cost function is updated
  		else:
  			toi=toimax+1
  		if(costglobal>cost):# after each toi-cycle, the value of cost function is compared with the best so far (costglobal)
  			costglobal=cost# if improvement is achieved the found r is set as rglobal and costglobal is updated
  			rglobal=r
  t += 1	
  np.savetxt('rglobal.csv', rglobal,delimiter=',',fmt='%i') # after tmax cycles the best clustering is rglobal






  	    
