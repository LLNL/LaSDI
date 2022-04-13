import numpy as np
from scipy import sparse as sp


def create_mask_1d(m,b,db):
  M2 = b + db*(m-1)
  mask = np.zeros((m,M2),dtype='int')
  
  block = np.ones(b,dtype='int')
  ind = np.arange(b)
  for row in range(m):
    col = ind + row*db
    mask[row,col] = block
  
  print("Sparsity in {} by {} mask: {:.2f}%".format(  m, 
                                                      M2, 
                                                      (1.0-np.count_nonzero(mask)/mask.size)*100 ) )
#  plt.figure()
#  plt.spy(mask)
#  plt.show()

  return mask

def create_mask_2d(nx,ny,m,b,db):
    
  # local
  Mb=sp.diags([np.ones(nx-2),np.ones(nx-2),np.ones(nx-2)],[0,-1,1],(nx-2,nx-2))
  M=sp.kron(sp.eye(ny-2),Mb,format="csr")

  Ib=sp.eye(nx-2)
  N=sp.kron(sp.diags([np.ones(ny-2),np.ones(ny-2),np.ones(ny-2)],[0,-1,1],(ny-2,ny-2)),Ib,format="csr")

  local=(M+N).astype('int8')
  I,J,V=sp.find(local)
  local[I,J]=1
  
#  col_ind=np.array([],dtype='int')
#  row_ind=np.array([],dtype='int')
#
#  for lin_ind in range(m):
#      j,i=np.unravel_index(lin_ind,(ny-2,nx-2))
#
#      E=np.ravel_multi_index((j,np.max((i-1,0))),(ny-2,nx-2))
#      W=np.ravel_multi_index((j,np.min((i+1,nx-2-1))),(ny-2,nx-2))
#      S=np.ravel_multi_index((np.max((j-1,0)),i),(ny-2,nx-2))
#      N=np.ravel_multi_index((np.min((j+1,ny-2-1)),i),(ny-2,nx-2))
#
#      col=np.unique([lin_ind,E,W,S,N])
#      row=lin_ind*np.ones(col.size,dtype='int')
#
#      col_ind=np.append(col_ind,col)
#      row_ind=np.append(row_ind,row)
#
#  data=np.ones(row_ind.size,dtype='int')
#  local2=sp.csr_matrix((data,(row_ind,col_ind)),shape=(m,m))

  # basis
  M2 = int(b + db*(m-1))
  basis = np.zeros((m,M2),dtype='int8')

  block = np.ones(b,dtype='int8')
  ind = np.arange(b)
  for row in range(m):
    col = ind + row*db
    basis[row,col] = block
  
  # mask
  col_ind=np.array([],dtype='int8')
  row_ind=np.array([],dtype='int8')
  for i in range(m):
    col=basis[sp.find(local[i])[1]].sum(axis=0).nonzero()[0]
    row=i*np.ones(col.size)

    col_ind=np.append(col_ind,col)
    row_ind=np.append(row_ind,row)

  data=np.ones(row_ind.size,dtype='int8')
  mask=sp.csr_matrix((data,(row_ind,col_ind)),shape=(m,M2)).toarray()
  
  print("Sparsity in {} by {} mask: {:.2f}%".format(m, M2, (1.0-np.count_nonzero(mask)/np.prod(mask.shape))*100))
  
  return mask

