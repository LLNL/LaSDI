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
  Mb=sp.diags([np.ones(nx),np.ones(nx),np.ones(nx)],[0,-1,1],(nx,nx))
  M=sp.kron(sp.eye(ny),Mb,format="csr")

  Ib=sp.eye(nx)
  N=sp.kron(sp.diags([np.ones(ny),np.ones(ny),np.ones(ny)],[0,-1,1],(ny,ny)),Ib,format="csr")

  local=(M+N).astype('int8')
  I,J,V=sp.find(local)
  local[I,J]=1
  
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

