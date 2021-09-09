def _pf_gas_conc(self, pos_x, pos_y, source_x, source_y, source_q): # true gas conectration
    avoid_zero = (np.sqrt(pow(source_x - pos_x,2) + pow(source_y - pos_y,2) ) < 1e-50) 
    source_x[avoid_zero] += 1e-50
    source_y[avoid_zero] += 1e-50

    dist = np.sqrt(pow((source_x - pos_x), 2) + pow(source_y - pos_y, 2))
    y_n = -(pos_x - source_x)*np.sin(self.wind_d*math.pi/180)+ \
           (pos_y - source_y)*math.cos(self.wind_d*math.pi/180)
    lambda_plume = math.sqrt(self.gas_d * self.gas_t / (1 + pow(self.wind_s,2) * self.gas_t/4/self.gas_d) )
    conc_com_1 = source_q/(4 * math.pi * self.gas_d * dist) 
    conc_com_2 = np.exp( -y_n * self.wind_mean_speed/(2*self.gas_d) - dist/lambda_plume)
    conc = conc_com_1 * conc_com_2
    return conc

def _weight_calculate(self, pf_x, pf_y, pf_q):
    pf_conc = self._pf_gas_conc(self.agent_x, self.agent_y, pf_x, pf_y, pf_q)
    #mean_pf_conc = sum(pf_conc*self.Wpnorms)
    #mean_conc = (mean_pf_conc + self.gas_measure)/2
    mean_conc = (pf_conc + self.gas_measure)/2
    pdetSig = np.sqrt( pow((mean_conc*self.sensor_sig_m),2) + pow(self.env_sig,2) )
    #if pdetSig < 1e-100: pdetSig = 1e-100
    pdetSig[pdetSig < 1e-100] = 1e-100
    pdetSig_sq = pow(pdetSig, 2)
    gauss_val = (self.gas_measure - pf_conc)/pdetSig
    gauss_new = 1/(math.sqrt(2*math.pi)*pdetSig_sq)*np.exp(-pow(gauss_val,2)/2)
    gauss_new[gauss_new < 1e-200] = 1e-200
    gauss_new[gauss_new != gauss_new] = 1e-200
    return gauss_new
    
def _particle_resample(self, gauss_new):
    N = self.Wpnorms.size
    M = N
    indx = np.ones(N)*-1
    Q = np.cumsum(self.Wpnorms)
    indx = np.zeros(N)
    T = np.arange(N)/N + self.np_random.uniform(low=np.zeros(N), high=np.ones(N)/N)
    i=0
    j=0
    while(i<N and j<M):
        while(Q[j] < T[i]):
            j = j+1
        indx[i]=j
        i=i+1

    indx = np.int64(indx)
    
    self.pf_x = self.pf_x[indx]
    self.pf_y = self.pf_y[indx]
    self.pf_q = self.pf_q[indx]

    mm = 2
    A=pow(4/(mm+2), 1/(mm+4) )
    cx = 4*math.pi/3
    hopt = A*pow(A,-1/(mm+4))
    for _ in range(1):
        CovXxp = np.var(self.pf_x)
        CovXyp = np.var(self.pf_y)
        CovXqp = np.var(self.pf_q)

        dkXxp = math.sqrt(CovXxp)+0.5
        dkXyp = math.sqrt(CovXyp)+0.5
        dkXqp = math.sqrt(CovXqp)+0.5

        nXxp = self.pf_x + (hopt*dkXxp*np.random.normal(0,1,self.pf_num) )
        nXxp[nXxp>self.court_lx] = self.court_lx # out of area
        nXxp[nXxp<0] = 0 # out of area

        nXyp = self.pf_y + (hopt*dkXyp*np.random.normal(0,1,self.pf_num) )
        nXyp[nXyp>self.court_ly] = self.court_ly # out of area
        nXyp[nXyp<0] = 0 # out of area

        nXqp = self.pf_q + (hopt*dkXqp*np.random.normal(0,1,self.pf_num) )
        nXqp[nXqp<0] = 0 # out of range
        nXqp[nXqp>self.q_max] = self.q_max

        n_new = self._weight_calculate(nXxp, nXyp, nXqp)
        alpha = n_new/gauss_new[indx]
        mcrand = np.random.uniform(0,1,self.pf_num)
#        print(alpha > mcrand)
        new_point_bool = alpha > mcrand
        self.pf_x[new_point_bool] = nXxp[new_point_bool]
        self.pf_y[new_point_bool] = nXyp[new_point_bool]
        self.pf_q[new_point_bool] = nXqp[new_point_bool]
    self.Wpnorms = np.ones(self.pf_num)/self.pf_num
    

def _particle_filter(self):
    self.update_count += 1
    Wp_sum = 0
    resample_true = False
    gauss_new = self._weight_calculate(self.pf_x, self.pf_y, self.pf_q)
#    if (gauss_new == np.ones(self.pf_num)/self.pf_num).all(): resample_true = True
    sort_g = np.sort(gauss_new)
    if (sort_g[self.pf_num-1] == sort_g[0] or self.update_count == 10): resample_true = True
    self.Wps = self.Wpnorms * gauss_new
    Wp_sum = np.sum(self.Wps)

    self.Wpnorms = self.Wps/Wp_sum

    if 1/sum(pow(self.Wpnorms,2)) < self.pf_num*0.5 or resample_true: # 1 for every time
        self.update_count = 0
        self._particle_resample(gauss_new)
        self.CovXxp = np.var(self.pf_x)
        self.CovXyp = np.var(self.pf_y)

