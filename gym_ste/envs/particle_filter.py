import psutil
import ray
import numpy as np
import math

#num_cpus = psutil.cpu_count(logical=False)
#ray.init(num_cpus=num_cpus)


def pf_gas_conc(args, i): # true gas conectration
    pos_x = args.agent_x
    pos_y = args.agent_y
    source_x = args.pf_x[i]
    source_y = args.pf_y[i]
    source_q = args.pf_q[i]
    if source_x == pos_x and source_y == pos_y: # to avoid divide by 0
        pos_x += 1e-3
        pos_y += 1e-3
    dist = math.sqrt(pow((source_x - pos_x), 2) + pow(source_y - pos_y, 2))
    y_n = -(pos_x - source_x)*math.sin(args.wind_mean_phi*math.pi/180)+ \
           (pos_y - source_y)*math.cos(args.wind_mean_phi*math.pi/180)
    lambda_plume = math.sqrt(args.gas_d * args.gas_t / (1 + pow(args.wind_mean_speed,2) * args.gas_t/4/args.gas_d) )
    conc = source_q/(4 * math.pi * args.gas_d * dist) * np.exp(-y_n * args.wind_mean_speed/(2*args.gas_d) - dist/lambda_plume)
    return conc

@ray.remote
def weight_update(args, i):
    pf_conc = pf_gas_conc(args, i)
    pdetSig = math.sqrt( pow((args.gas_measure*args.sensor_sig_m),2) + pow(args.env_sig,2) )
    pdetSig_sq = pow(pdetSig, 2)
    if pdetSig_sq < 1e-100:
        pdetSig_sq = 1e-100

    p_val = (args.gas_measure - pf_conc)/pdetSig
    p_new = 1/(math.sqrt(2*math.pi)*pdetSig_sq)*np.exp(-pow(p_val,2)/2)
    if p_new == 0:
        p_new = 1e-100

    return p_new


class ParticleFilter:
    def __init__(self, args):
#        num_cpus = psutil.cpu_count(logical=False)
 #       ray.init(num_cpus=num_cpus)

        self.agent_x = args.agent_x
        self.agent_y = args.agent_y
        self.pf_x = args.pf_x
        self.pf_y = args.pf_y
        self.pf_q = args.pf_q
        self.gas_measure = args.gas_measure
        self.sensor_sig_m = args.sensor_sig_m
        self.env_sig = args.env_sig
        self.Wpnorms = args.Wpnorms
        self.np_random = args.np_random
        
        self.court_lx = args.court_lx
        self.court_ly = args.court_ly
        self.pf_num = args.pf_num
        self.Wps = args.Wps
        
        self.wind_mean_phi = args.wind_mean_phi
        self.wind_mean_speed = args.wind_mean_speed
        self.gas_d = args.gas_d
        self.gas_t = args.gas_t
        
        self._particle_filter()
        

    def _resample(self, gauss_new):
        N = self.Wpnorms.size
        M = N
        indx = np.ones(N)*-1
        Q = np.cumsum(self.Wpnorms)
        indx = np.zeros(N)
        T = np.arange(N)/N + self.np_random.uniform(low=np.zeros(N), high=np.ones(N)/N)
        i=1
        j=1
        while(i<N and j<M):
            while(Q[j] < T[i]):
                j = j+1
            indx[i]=j
            i=i+1

        indx = np.int64(indx)
        for i in range(0,N):
            self.pf_x[i] = self.pf_x[indx[i]]
            self.pf_y[i] = self.pf_y[indx[i]]
            self.pf_q[i] = self.pf_q[indx[i]]

        mm = 2
        A=pow(4/(mm+2), 1/(mm+4) )
        cx = 4*math.pi/3
        hopt = A*pow(A,-1/(mm+4))
        for _ in range(3):
            CovXxp = np.var(self.pf_x)
            CovXyp = np.var(self.pf_y)
            CovXqp = np.var(self.pf_q)

            dkXxp = math.sqrt(CovXxp)+0.01
            dkXyp = math.sqrt(CovXyp)+0.01
            dkXqp = math.sqrt(CovXqp)+0.01
            nXxp = self.pf_x + (hopt*dkXxp*np.random.normal(0,1,self.pf_num) )
            nXxp[nXxp>self.court_lx] = self.court_lx # out of area
            nXxp[nXxp<0] = 0 # out of area
            nXyp = self.pf_y + (hopt*dkXyp*np.random.normal(0,1,self.pf_num) )
            nXyp[nXyp>self.court_ly] = self.court_ly # out of area
            nXyp[nXyp<0] = 0 # out of area
            nXqp = self.pf_q + (hopt*dkXqp*np.random.normal(0,1,self.pf_num) )
            nXqp[nXqp<0] = 0 # out of range

            n_new = [weight_update.remote(self,i) for i in range(0,N)]
            n_new = ray.get(n_new)
            for i in range(0,N):
                # n_new.append(weight_update(i))
                alpha = n_new[i]/gauss_new[indx[i]]
                mcrand = np.random.uniform(0,1,1)
                if alpha > mcrand:
                    self.pf_x[i] = nXxp[i]
                    self.pf_y[i] = nXyp[i]
                    self.pf_q[i] = nXqp[i]

        self.Wpnorms = np.ones(self.pf_num)/self.pf_num


    def _particle_filter(self):
        pf_concs = []
        Wp_sum = 0
        gauss_new = [weight_update.remote(self,i) for i in range(0,self.pf_num)]
        gauss_new = ray.get(gauss_new)
        for i in range(0,self.pf_num):
            Wpnorm = self.Wpnorms[i]
            Wp = Wpnorm * gauss_new[i]
            self.Wps[i] = Wp
            Wp_sum += Wp

        self.Wpnorms = self.Wps/Wp_sum
        if 1/sum(pow(self.Wpnorms,2)) < self.pf_num*0.5: # 1 for every time
#            print("-------------------------------------")
#            print("resample")
#            print("-------------------------------------")
            self._resample(gauss_new)

        return [np.array(self.pf_x), np.array(self.pf_y), np.array(self.pf_q), np.array(self.Wps), np.array(self.Wpnorms)]
