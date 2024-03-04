import h5py
from funcs import *

freezing_pts = jn_zeros(0, 5)


Ns = [4,6,7]
ws = np.linspace(0.1,20, 50)
beta = float('inf')
sx,sy,sz = sigmax(), sigmay(), sigmaz()

periods = 10


pos1 = 0
pos2 = int(N/2)-1

for N in tqdm(Ns):
    nprocs = 2
    p = Pool(processes = nprocs) 
    opts = Options(nsteps=1e5, num_cpus=1, openmp_threads=1)

    for i,xx in enumerate(jn_zeros(0,10)):
        ax1.axvline(x=xx, alpha=0.25, color = 'black', linestyle = '--')

    i,j = pos1, pos2
    id = qeye(2**i)
    dim11 = j-i-1
    id1 = qeye(2**dim11)
    dim12 = N-1-j
    id2 = qeye(2**dim12)
    sz_cor = Qobj(tensor(id, tensor(sz, tensor(id1, tensor(sz,id2)))).full()) 

    param = [{'h0':0, 'h':jn_zeros(0,10)[9] * w/2, 'N':N,'opts':opts, 'sz_cor':sz_cor,'w':w,\
               'beta':beta,'times':np.linspace(0, periods * 2*np.pi/w, 100, endpoint=False)} for w in ws]

    datas = np.array(p.map(run_dynm_corr_avg,param))
    
    fname = "sz"+str(pos1)+"sz"+str(pos1)+"N_"+ str(N)+"_.hdf5"

    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('szsz', np.shape(datas), data=datas)
        hf.create_dataset('ws', np.shape(ws), data=ws)
        hf.attrs['N'] = N  
        hf.attrs['pos1'] = pos1
        hf.attrs['pos2'] = pos2
        hf.attrs['beta'] = beta