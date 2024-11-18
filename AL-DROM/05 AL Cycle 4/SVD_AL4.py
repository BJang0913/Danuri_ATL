import numpy as np
import math
import os 
import time


def main():

    n_modes = 100    # The number of training cases
    n_snaps = 100    # The number of training cases
    energyThreshold = 0.995    # 99.5% energy from original matrix using singular value accumulation

    scalar = 'T'

    print(time.ctime())
    stime=time.time()
      
    svd_scalar(n_modes,n_snaps,scalar,energyThreshold)
    print(time.ctime())
    print("cpu time : ",  time.time()- stime)

def svd_scalar(n_modes,n_snaps,scalar,energyThreshold):
    
    snap_path = 'Training_output/14OD{0}.txt'
    snap_inp = open(snap_path.format(1),'r')
    snap_inp_lines = snap_inp.readlines()
    int_length = len(snap_inp_lines)

    snap_stack = np.zeros((int_length,n_snaps))

    for t in range(n_snaps):
        snap_inp = open(snap_path.format((t+1),scalar),'r')
        snap_inp_lines = snap_inp.readlines()

        for j in range(int_length):
            snap_stack[j,t] = float(snap_inp_lines[j])


    U,s,V = np.linalg.svd(snap_stack,full_matrices = False)

    print(U.shape)
    print('Orthogonality check :', np.inner(U[:,0],U[:,1]))

    energy = s / np.sum(s)
    print(s)
    np.savetxt("sigma.txt", s)
    totalEnergy = 0
    for i in range(n_modes):
        totalEnergy += energy[i]
        if totalEnergy > energyThreshold:
            truncId = i+1
            break
            
    print('We are using {0} modes for conserving {1}% of original {2} data!'.format(truncId,energyThreshold*100,scalar))
    
    
    if not os.path.isdir('POD_initial'):
        os.mkdir('POD_initial')

    for i in range(n_modes):
        if not os.path.isdir('POD_initial/{0}'.format(i+1)):
            os.mkdir('POD_initial/{0}'.format(i+1))
        
    
        pod_write = open('POD_initial/{0}/{1}'.format(i+1,scalar),'w')

        for j in range(int_length):
            pod_j = U[j,i]
            pod_write.write(str(pod_j))
            pod_write.write('\n')
            
        pod_write.close()

if __name__ == "__main__":
    main()

