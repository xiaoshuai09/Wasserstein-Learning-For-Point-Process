import numpy as np
import scipy.stats
import matplotlib
from _struct import Struct
matplotlib.use('agg')
import matplotlib.pyplot as plt

def read_sequences(seq_file):
    with open(seq_file) as f:
        lines = f.readlines()
    sequences = []
    for line in lines:
        x = line.split()
        seq = [None]*len(x)
        for i in range(len(x)):
            seq[i] = float(x[i])
        sequences.append(seq)
    f.close()
    return sequences

def  get_intensity(sequences, T = None, n_t=None, t0=None):
    if T is None:
        T = max(max(sequences))
    if n_t is None:
        n_t = 50
    if t0 is None:
        t0 = 0

    dt = (T-t0)/n_t
    ts = np.arange(t0,T,dt)
    n_seqs = len(sequences)
    lens = np.zeros((n_seqs,1))
    cnt = np.zeros((n_t,1))

    for i in range(n_seqs):
        seq = sequences[i]
        j = 0
        k = 0
        for t in np.arange(t0+dt,T+dt,dt):
            while (j < len(seq) and seq[j] <= t):
                j = j + 1
            cnt[k] = cnt[k] + j
            k = k + 1

            #print(t)
    dif = np.zeros((len(cnt),1))
    dif[0] = cnt[0]
    for i in range(len(cnt)-1):
        dif[i+1] = cnt[i+1]-cnt[i]
    intensity = dif/(n_seqs)/dt

    return ts, intensity


def get_integral_empirical(sequences, intensity, T, n_t, t0=None):

    if T is None:
        T = max(max(sequences))
    if n_t is None:
        n_t = 50
    if t0 is None:
        t0 = 0
    dt = (T-t0)/n_t
    ts = np.arange(t0,T,dt)
    n_seqs = len(sequences)

    
    integral = []
    for i in range(1000):
        seq = sequences[i]
        integral_seq = []
        for j in range(len(seq)-1):
            t_start = seq[j]
            t_end = seq[j+1]
            index_start = np.int( t_start/dt)
            index_end = np.int(t_end/dt)+1
            integral_seq.append( np.sum(intensity[index_start:index_end])*dt -intensity[index_start]*(t_start-index_start*dt)-intensity[index_end-1]*(index_end*dt-t_end))
            #-intensity[index_start]*(t_start-index_start*dt)-intensity[index_end-1]*(index_end*dt-t_end)
        
        integral += integral_seq
    return integral

  
def hawkes_integral(sequences,model):
    integrals = []
    for seq in sequences:
        integral = []
        seq = np.asarray(seq)
        for i in range(len(seq)-1):
            integral_delta = (seq[i+1]-seq[i])*model['mu'] + model['alpha'] * np.sum(np.exp(-(seq[i]-seq[:i+1]))-np.exp(-(seq[i+1]-seq[:i+1])))
            integral.append(integral_delta)
        integrals+=integral
    return integrals

def selfcorrecting_integral(sequences,model):
    integrals = []
    for seq in sequences:
        integral = []
        seq = np.asarray(seq)
        for i in range(len(seq)-1):
            integral_delta = (np.exp(model['mu']*seq[i+1]) - np.exp(model['mu']*seq[i]))/np.exp(model['alpha']*len(seq[:i+1]))/model['mu']
            integral.append(integral_delta)
        integrals+=integral
    return integrals

def gaussian_integral(sequences,model):
    integrals = []
    for seq in sequences:
        integral = []
        seq = np.asarray(seq)
        for i in range(len(seq)-1):
            integral_delta = np.sum( model['coef'] * (scipy.stats.norm.cdf(seq[i+1], model['center'], model['std']) - scipy.stats.norm.cdf(seq[i], model['center'], model['std']) ) )
            integral.append(integral_delta)
        integrals+=integral
    return integrals

def get_integral(sequences, data):
    sequences = sequences[:1000] # random sample?
    if data=='hawkes':
          model = dict()
          model['mu'] = 1
          model['w'] = 1
          model['alpha'] = 0.8
          integrals = hawkes_integral(sequences, model)
    elif data=='selfcorrecting':
          model = dict()
          model['mu'] = 1
          model['alpha'] = 0.2
          integrals = selfcorrecting_integral(sequences, model)
    elif data=='gaussian':
          model = dict()
          model['coef'] = [2,3,2]
          model['center'] = [3,7,11]
          model['std'] = [1,1,1]
          integrals = gaussian_integral(sequences, model)
    return integrals
    
if __name__ == '__main__':
    seq_file = 'real_data_sequences.txt'
    sequences = read_sequences(seq_file)
    
    seq_file_gen = 'real_data_sequences.txt'
    sequences_gen = read_sequences(seq_file_gen)
    
    T = 800
    n_t  = 30
    ts, intensity = get_intensity(sequences, T, n_t)
    ts_gen, intensity_gen = get_intensity(sequences_gen, T, n_t)
    
    plt.plot(ts,intensity, label='real')
    plt.plot(ts_gen, intensity_gen, label='generated')
    plt.legend(loc=1)
    plt.xlabel('time')
    plt.ylabel('intensity')
