from qutip import mesolve, basis, jmat, Options, expect, Qobj
from multiprocessing import Pool
from tqdm import tqdm
from scipy.special import jn_zeros
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (12,7),"font.size": 13})

def drive(t, args):
    w = args['omega']
    h = args['h']
    h0 = args['h0']
    return h * np.sin(w*t) + h0

def get_hamiltonians(N):
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    kn =  2.0/(N-1)                                     
    H0 = kn * sz **2 
    H1 = 2 * sx
    return H0,H1

def sys_evolve(w,N,opts):
    sx,sy,sz = jmat(N,"x"),jmat(N,"y"),jmat(N,"z")
    T = 2 * np.pi/w
    h = jn_zeros(0,5)[0]* w / 4.0
    params = {'h0':np.pi/32, 'h':h,'omega':w, 'N':N} 
    H0, H1 = get_hamiltonians(N)
    H = [H0,[H1,drive]]
    en, sts = sx.eigenstates() 
    rho0 = sts[-1]
    times = np.linspace(0,100*T, 100)
    hbar = []
    out = mesolve(H, rho0, times, [], [H1/(2*N)], args = params, options=opts)
    return np.average(out.expect), (times/(2*np.pi/w)).reshape(len(times),1)

if __name__ == '__main__':
    nprocs = 32
    
    o1 = np.linspace(0.29, 1.58, 15)
    o2 = np.linspace(1.6, 2.5, 15)
    o3 = np.linspace(2.8, 4.38, 15)
    o4 = np.linspace(4.5, 9.4, 15)
    o5 = np.linspace(9.5, 20.0, 15)
    o6 = np.linspace(20.1, 40.0, 5)
    
    o7 = np.linspace(0.1, 2, 15)
    o8 = np.linspace(1.6, 2.5, 15)
    o9 = np.linspace(2.8, 4.38, 15)
    
    o10 = np.linspace(25, 90, 10)


    omega_vals = np.concatenate((o1, o2, o3, o4, o5))
    #omega_vals = np.concatenate((o7,o8,o9))
    #omega_vals = o10
    
    
    Ns = [100]
    nstp = [1e5]
    for n, N in enumerate(Ns):
        p = Pool(processes = nprocs)
        print("running for TSS spin N=",N, 'nprocs=',nprocs," !")
        opts = Options(nsteps=nstp[n], num_cpus=nprocs, openmp_threads=1, atol=1e-12, rtol=1e-14)
        data = np.array(p.starmap(sys_evolve,tqdm([(w,N, opts) for w in omega_vals])))
        
        sxavg = []
        for i in range(len(omega_vals)):
            time = np.array(data[i][1])[:,0]
            sxavg.append(data[i][0])
        lbl = 'N=' + str(N)
        plt.plot(omega_vals, sxavg, label = lbl)
    plt.legend(frameon =False)    
    plt.tick_params(which='both', axis="x", direction="in", width= 1.0)
    plt.tick_params(which='both', axis="y", direction="in", width= 1.0)
    plt.ylim(-1.1,1.1)
    plt.xlabel('time')
    plt.ylabel('<<Sx>>')
    plt.savefig('', dpi= 500)
    plt.show()