import numpy as np

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


