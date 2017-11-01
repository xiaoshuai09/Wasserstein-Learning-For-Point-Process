import numpy as np
import os

def sequence_filter(sequences,seqlen=None,T=None):
    if T==None:
        reduced = [None]*len(sequences)
        for i,seq in enumerate(sequences):
            reduced[i] = seq[:seqlen[i]]
    else:
        reduced = []
        for seq in sequences:
            line = seq[seq<=T]
            if len(line)>0:
                reduced.append(line)
    return reduced
  
  
def file2sequence(filename):
    sequences = []
    if os.path.isfile('data/{}.txt'.format(filename)):
        f = open('data/{}.txt'.format(filename)) 
    elif os.path.isfile('/nv/hcoc1/sxiao40/data/code/MultiVariatePointProcess-master/example/data/{}.txt'.format(filename)):
        f = open('/nv/hcoc1/sxiao40/data/code/MultiVariatePointProcess-master/example/data/{}.txt'.format(filename)) 
    else:
        print filename
        raise Exception("File doesn't exist.")
 
    for line in f:
      line = line.strip()
      if line:
          seq=line.split('\t')
          seq = [float(item) for item in seq]
          if seq:
              sequences.append(seq)
      else:
          #print ('this line have no sequence')
          pass
    f.close()
    return sequences

def sequence2file(sequences,filename):
    with open('data/{}.txt'.format(filename),'wb') as f:
        for line in sequences:
          if len(line)>0:
              for it in line:
                  f.write("{}\t".format(it))
              f.write("\n")

def lambda_estimation(sequences,num_dim,T):
    estimated_lambda = np.zeros(num_dim)
    for seq in sequences:
        for item in seq:
          estimated_lambda[np.int16(item[1])]+=1
    
    estimated_lambda/=(len(sequences)*T)
    return estimated_lambda
  

def dimension_extract(sequences,num_dim,T):
    estimated_lambda = []
    for seq in sequences:
        for dim in range(num_dim):
            seq_dim = filter(lambda x:x[1]==dim,seq)
            if seq_dim:
                pass

    return estimated_lambda