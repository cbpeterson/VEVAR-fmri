import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from scipy import sparse
import scipy.io as sio
from scipy.linalg import block_diag
from scipy import stats

def update_u(pi,sig_q,sig_w,exp_phi_T_phi,Y,exp_phi,sum_others):
    return(np.log(pi/(1 - pi)) + 0.5*(np.log(sig_q/sig_w)) - 0.5*np.log(exp_phi_T_phi + sig_q/sig_w) + (1/(2*sig_q))*((Y.T.dot(exp_phi) - sum_others)**2)/(exp_phi_T_phi + (sig_q/sig_w)))

def update_mu_w(exp_phi,Y,sum_others,exp_phi_T_phi,sig_q,sig_w):
    return((exp_phi.T.dot(Y) - sum_others)/(exp_phi_T_phi + sig_q/sig_w))

def update_sig_w(sig_q,sig_w,exp_phi_T_phi):
    return(sig_q/(exp_phi_T_phi + sig_q/sig_w))

def update_SIGMA_phi(K,alpha,n):
    return(K.dot(np.linalg.inv(alpha*K + np.eye(n))))

def update_MU_phi(SIGMA_phi,alpha,Y,sum_other):
    return((SIGMA_phi*alpha).dot(Y - sum_other))

def expected_SxW(gamma_w,mu_w,sig_w):
    return(gamma_w*(mu_w**2 + sig_w))

def expected_phi_T_phi(MU_phi,SIGMA_phi):
    return(np.trace(SIGMA_phi) + MU_phi.T.dot(MU_phi))

def update_sig_q(Y,sum_all_1,sum_all_2,sum_all_3,n):
    mat = Y.dot(Y.T) - Y*sum_all_1 + sum_all_2 + 2.0*sum_all_3
    return(np.trace(mat)/n)

def update_sig(sum_exp,sum_gamma):
    return(sum_exp/sum_gamma)

def update_sigma_q(n,Y,sum_all,sum_all2,sum_greater):
    return((1/n)*np.trace(Y.dot(Y.T)) - Y.dot(sum_all) + sum_all2 + 2*sum_greater)

def update_subject_edge(Xi,U,Sig_inv,X,Omega):

    R = Xi.shape[0]
    blocks = extract_block_diag((np.kron(Xi,U.T.dot(U)) + Sig_inv),R)
    blocks_inv = [np.linalg.inv(b) for b in blocks]
    Sig_s = block_diag(*blocks_inv)

    Mu_s = Sig_s.dot(np.kron(Xi,U.T).dot(X) + Sig_inv.dot(Omega)) 
    return(Mu_s,Sig_s) 

def se(X,Y,l):
    set_to_zero = np.where(np.isnan(X))
    r = np.absolute(np.subtract.outer(X,Y))
    n = r.shape[0]
    kernel = np.exp((-(r*l)**2)/l)
    kernel[set_to_zero,:] = 0
    kernel[:,set_to_zero] = 0
    kernel[set_to_zero,set_to_zero] = 1.0
    return(kernel)

def linear(X,Y):
    return(np.outer(X,Y))

def log_prob_SE(par,*args):
    X = args[0][0]
    Y = args[0][1]
    w = par[0:(len(par)/2)]
    l = par[(len(par)/2):len(par)]
    K = se(X,X,l)
    n = K.shape[0]
    cc = K + np.eye(n,n)*0.0001
    r = -0.5 * np.log(np.linalg.det(cc)) - 0.5*Y.T.dot(np.linalg.inv(cc)).dot(Y) - (n/2) * (2*np.pi)
    return(-r)

def update_sig_q(y,F,G):

    return(np.trace(y[:,np.newaxis].dot(y[:,np.newaxis].T) - y[:,np.newaxis].dot(F[:,np.newaxis].T) + G) / S)

def group_order(X):
    xs = [0]*len(X)
    for i in range(len(X)):
        xs[i] = (X[:(i)] == X[i]).sum()
    return(xs)

def extract_block_diag(A,M,k=0):
    """Extracts blocks of size M from the kth diagonal
    of square matrix A, whose size must be a multiple of M."""

    # Check that the matrix can be block divided
    if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
        raise StandardError('Matrix must be square and a multiple of block size')

    # Assign indices for offset from main diagonal
    if abs(k) > M - 1:
        raise StandardError('kth diagonal does not exist in matrix')
    elif k > 0:
        ro = 0
        co = abs(k)*M 
    elif k < 0:
        ro = abs(k)*M
        co = 0
    else:
        ro = 0
        co = 0

    blocks = np.array([A[i+ro:i+ro+M,i+co:i+co+M] 
                       for i in range(0,len(A)-abs(k)*M,M)])
    return blocks


def VEVAR(X,COV,groups,L,pi_B,pi_phi,ls,convergence_tol,max_iters):
#############################################
# X = time x region x subject matrix
# COV = P x subject matrix   
# groups = vector with group assignments   
# L = integer with lag values to use
# pi_B = float with prior probability of edge
# pi_phi = float with prior probability of covariate selected
# ls = length scale of kernel function
# convergence_tol = when to stop algorithm
# max_iters = maximum number of iterations of algorithm before stopping
###############################################

########################
# warming initialization
########################

    global R,G,P
        
    X_in = np.copy(X)
    n = X.shape[0]
    R = X.shape[1]
    S = X.shape[2]
    P = COV.shape[0]
    G = len(np.unique(groups))
    Gs = group_order(groups)
    coefs = R*R*L
    Xs = np.reshape(X[1:,:,:],((n-1)*R,S),order = "F")
    Us = X[0:(n - 1),:,:]
    cov_nums = COV.shape[0]
    
    grps = []
    grp_sub_id = []
    subs = 0
    S_g = []
    for g in range(G):
        grp = np.where(groups == g)
        grp_sub_id += grp
        grps += [np.arange(0,len(grp[0]))]
        S_g += [len(grp[0])]
        
        
    ## hyperparameters
    # a_sig1, b_sig1: priors on variance of slab
    # a_sig0, b_sig0: priors on variance of spike
    a_sig1 = 2
    b_sig1 = 1
    a_sig0 = 2
    b_sig0 = 1

    sig_w = 1 
    sig_u = 1

    ## initialize
    Xi = np.tile(np.eye(R),(G,1,1))
    a_qs_1 = np.ones(G)
    b_qs_1 = np.ones(G)
    sig_qs_1 = np.ones(G)*0.005

    a_qs_0 = np.ones(G)
    b_qs_0 = np.ones(G)
    sig_qs_0 = np.ones(G)
    
    q_beta_mean = np.zeros((R*R*L,S)) # subject level mean
    q_beta_variance = np.ones((R*R*L,S)) # subject level variance

    # prob of edge selected, the initial values may need to be modified for real data
    gamma_B = np.ones((coefs,G)) * np.array([0.5, 0.5]) 
    
    k = np.ones((coefs,G)) # prob of edge selected
    Fi_mean = np.zeros((coefs,G)) # function mean
    Fi_var = np.ones((coefs,G))
    
    sig_ws = np.ones((coefs,G))*sig_w # variance of spike and slab per edge
    mu_ws = np.zeros((coefs,G)) # mean of spike and slab per edge
    
    n_k = 0

    MU_phis = [] # GP mean
    SIGMA_phis = [] # GP covariance
    B_g_mean = [] # edge function mean
    B_g_variance = [] # edge function variance
    for g in range(G):
        MU_phis += [np.zeros((P,coefs,S_g[g]))]
        SIGMA_phis += [np.tile(np.eye(S_g[g]),(P,coefs,1,1))]
        B_g_mean += [np.zeros((coefs,S_g[g]))]
        B_g_variance += [np.zeros((coefs,G,S_g[g],S_g[g]))]


    MU_init = []
    SIGMA_init = []
    for g in range(G):
        MU_init += [np.zeros((P,coefs,S_g[g]))]
        SIGMA_init += [np.tile(np.eye(S_g[g]),(P,coefs,1,1))]

    weight_var = np.ones((P,coefs,G)) # variance of weight for each covariate and edge
    weight_mean = np.zeros((P,coefs,G)) # mean of weight for each covariate and edge
    weight_prob = np.ones((P,coefs,G)) # probability of weight for each covariate and edge (gamma)
    
    wv_init = np.zeros((P,coefs,G))
    wm_init = np.zeros((P,coefs,G))
    wp_init = np.zeros((P,coefs,G))

    rand_order_groups = np.random.permutation(G)

    for g in range(G):
        rand_order_subjects = np.random.permutation(S_g[g])
        rand_order_coefs = np.random.permutation(coefs)
        rand_order_covariates = np.random.permutation(P)
        for s in rand_order_subjects:
            U = Us[:,:,grp_sub_id[g][s]]
            X = Xs[:,grp_sub_id[g][s]]
            q_beta_mean[:,grp_sub_id[g][s]], var = update_subject_edge(Xi[g,:,:],U,np.eye(R*R*L)*1,X,np.zeros(R*R*L))
            q_beta_variance[:,grp_sub_id[g][s]] = np.diag(var) 

        for r in rand_order_coefs:
            for p in rand_order_covariates:
                SIGMA_phis[g][p,r,:,:] = se(COV[p,grp_sub_id[g]],COV[p,grp_sub_id[g]],ls)
                MU_phis[g][p,r,:] = SIGMA_phis[g][p,r,:,:].dot(q_beta_mean[r,grp_sub_id[g]]*0.7 + np.random.normal(0,0.01,S_g[g]))
        
        
######## warming of the model: average across many random starts, allowing each covariate to be the first proposed ########        
        burns = 10
        
        for iter_P in range(burns):
            for p in np.random.permutation(P):
                weight_var[:,:,g] = np.ones((P,coefs)) # variance of weight for each covariate and edge
                weight_mean[:,:,g] = np.zeros((P,coefs)) # mean of weight for each covariate and edge
                weight_prob[:,:,g] = np.zeros((P,coefs)) # probability of weight for each covariate and edge (gamma)
                
                rand_order_covariates = np.random.permutation(P)
                swap = np.where(rand_order_covariates == p)[0].tolist()[0]
                rand_order_covariates[swap],rand_order_covariates[0] = rand_order_covariates[[0,swap]]
                for r in range(coefs):
                    Y = q_beta_mean[r,grp_sub_id[g]] - Fi_mean[r,g]
                    for p in rand_order_covariates:
                        exp_phi_T_phi = expected_phi_T_phi(MU_phis[g][p,r,:],SIGMA_phis[g][p,r,:,:])
                        sum_others = 0.0
                        for not_p in np.delete(rand_order_covariates,np.where(rand_order_covariates == p)):
                            sum_others += ((weight_prob[not_p,r,g]*weight_mean[not_p,r,g])*MU_phis[g][p,r,:][:,np.newaxis].T).dot(MU_phis[g][not_p,r,:].T) 
                        u_w = update_u(pi_phi,sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi,Y,MU_phis[g][p,r,:],sum_others)
                        weight_prob[p,r,g] = 1/(1 + np.exp(-u_w))
                        weight_mean[p,r,g] = update_mu_w(MU_phis[g][p,r,:],Y,sum_others,exp_phi_T_phi,sig_qs_1[g],sig_ws[r,g]) ## check
                        weight_var[p,r,g] = update_sig_w(sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi)
                wv_init[:,:,g] += weight_var[:,:,g]
                wm_init[:,:,g] += weight_mean[:,:,g]
                wp_init[:,:,g] += weight_prob[:,:,g]

        for iter_P in range(burns):
            for p in np.random.permutation(P):
                weight_var[:,:,g] = np.ones((P,coefs)) # variance of weight for each covariate and edge
                weight_mean[:,:,g] = np.zeros((P,coefs)) # mean of weight for each covariate and edge
                weight_prob[:,:,g] = np.zeros((P,coefs)) # probability of weight for each covariate and edge (gamma)        
        
                rand_order_covariates = np.random.permutation(P)
                swap = np.where(rand_order_covariates == p)[0].tolist()[0]
                rand_order_covariates[swap],rand_order_covariates[P-1] = rand_order_covariates[[P-1,swap]]
                for r in range(coefs):
                    Y = q_beta_mean[r,grp_sub_id[g]] - Fi_mean[r,g]
                    for p in rand_order_covariates:
                        exp_phi_T_phi = expected_phi_T_phi(MU_phis[g][p,r,:],SIGMA_phis[g][p,r,:,:])
                        if exp_phi_T_phi < 0:
                            print("exp_phi_T_phi < 0 ","r = ",r," p = ",p)
                        elif sig_qs_1[g] < 0:
                            print("sig_qs is the problem! R = ",r," p = ",p)
                        sum_others = 0.0
                        for not_p in np.delete(rand_order_covariates,np.where(rand_order_covariates == p)):
                            sum_others += ((weight_prob[not_p,r,g]*weight_mean[not_p,r,g])*MU_phis[g][p,r,:][:,np.newaxis].T).dot(MU_phis[g][not_p,r,:].T) 
                        u_w = update_u(pi_phi,sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi,Y,MU_phis[g][p,r,:],sum_others)
                        weight_prob[p,r,g] = 1/(1 + np.exp(-u_w))
                        weight_mean[p,r,g] = update_mu_w(MU_phis[g][p,r,:],Y,sum_others,exp_phi_T_phi,sig_qs_1[g],sig_ws[r,g]) ## check
                        weight_var[p,r,g] = update_sig_w(sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi)
                wv_init[:,:,g] += weight_var[:,:,g]
                wm_init[:,:,g] += weight_mean[:,:,g]
                wp_init[:,:,g] += weight_prob[:,:,g]
        weight_prob[:,:,g] = wp_init[:,:,g] / (2*P*burns)
        weight_mean[:,:,g] = wm_init[:,:,g] / (2*P*burns)
        weight_var[:,:,g] = wv_init[:,:,g] / (2*P*burns)
        

        for r in rand_order_coefs:
            for p in rand_order_covariates:
                exp_w = weight_mean[p,r,g]*weight_prob[p,r,g]
                B_g_mean[g][r,:] += MU_phis[g][p,r,:]*exp_w
                
        for r in rand_order_coefs:
            Fi_var[r,g] = 1/(S_g[g]/sig_qs_1[g] + 1/sig_u)
            Fi_mean[r,g] = Fi_var[r,g] * (np.sum(q_beta_mean[r,grp_sub_id[g]] - np.sum(MU_phis[g][:,r,:].T*weight_mean[:,r,g]*weight_prob[:,r,g],1))/sig_qs_1[g])

    # Fi_mean = np.zeros((coefs,G))

    start_time = time.time()
    # print(weight_prob)
    print("starting algorithm")
    iters = 0
    max_change = convergence_tol * 100
    
    rand_subs = np.random.permutation(S)[:10]
    rand_edges = np.random.permutation(R*R*L)[:25]
    rand_ROI = np.random.permutation(R)[:5]
    ELBO = 0
    while ((max_change > convergence_tol) & (iters < max_iters)):
        iters += 1
        n_k += 1
    
        ## copy stored variables to keep track of amount changed ##
        weight_prob_copy = weight_prob.copy()
        weight_mean_copy = weight_mean.copy()
        weight_var_copy = weight_var.copy()
        q_beta_copy = q_beta_mean.copy()
        MU_copy = MU_phis[:]
        SIGMA_copy = SIGMA_phis[:]
        gamma_B_copy = gamma_B.copy()
        sig_0_copy = sig_qs_0.copy()
        sig_1_copy = sig_qs_1.copy()
        Fi_copy = Fi_mean.copy()
        k_copy = k.copy()
        
        rand_order_groups = np.random.permutation(G)
        
        for g in rand_order_groups:
            rand_order_coefs = np.random.permutation(coefs)
            rand_order_covariates = np.random.permutation(P)
            rand_order_subjects = np.random.permutation(S_g[g])

            st = time.time()
            ### update subject level edges
            sig_inv = (np.diag(gamma_B[:,g]*(1.0/sig_qs_1[g]) + (1-gamma_B[:,g])*(1.0/sig_qs_0[g])))
            Xi_inv = np.linalg.inv(Xi[g,:,:])

            for s in rand_order_subjects:
                U = Us[:,:,grp_sub_id[g][s]]
                X = Xs[:,grp_sub_id[g][s]]
                q_beta_mean[:,grp_sub_id[g][s]], var = update_subject_edge(Xi_inv,U,sig_inv,X,B_g_mean[g][:,s])
                q_beta_variance[:,grp_sub_id[g][s]] = np.diag(var)
                
            for r in rand_order_coefs:
                Fi_var[r,g] = 1/(S_g[g]/sig_qs_1[g] + 1/sig_u)
                Fi_mean[r,g] = Fi_var[r,g] * (np.sum(q_beta_mean[r,grp_sub_id[g]] - np.sum(MU_phis[g][:,r,:].T*weight_mean[:,r,g]*weight_prob[:,r,g],1))/sig_qs_1[g]) 
            
            
            st = time.time()
            ### Update Spike and Slab elements for each edge and coviariate
            for r in rand_order_coefs:
                Y = q_beta_mean[r,grp_sub_id[g]] - Fi_mean[r,g]
                for p in rand_order_covariates:
                    exp_phi_T_phi = expected_phi_T_phi(MU_phis[g][p,r,:],SIGMA_phis[g][p,r,:,:])
                    sum_others = 0.0
                    for not_p in np.delete(rand_order_covariates,np.where(rand_order_covariates == p)):
                        sum_others += ((weight_prob[not_p,r,g]*weight_mean[not_p,r,g])*MU_phis[g][p,r,:][:,np.newaxis].T).dot(MU_phis[g][not_p,r,:].T) 
                    u_w = update_u(pi_phi,sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi,Y,MU_phis[g][p,r,:],sum_others)
                    weight_prob[p,r,g] = 1/(1 + np.exp(-u_w))
                    weight_mean[p,r,g] = update_mu_w(MU_phis[g][p,r,:],Y,sum_others,exp_phi_T_phi,sig_qs_1[g],sig_ws[r,g]) ## check
                    weight_var[p,r,g] = update_sig_w(sig_qs_1[g],sig_ws[r,g],exp_phi_T_phi)
                
            ### Update each GP for edge and covariate
            st = time.time()
            for r in rand_order_coefs:
                Y = q_beta_mean[r,grp_sub_id[g]] - Fi_mean[r,g]
                for p in rand_order_covariates:
                    K = se(COV[p,grp_sub_id[g]],COV[p,grp_sub_id[g]],ls)
                    sum_other = 0
                    for not_p in np.delete(range(P),p):
                        sum_other += ((weight_prob[not_p,r,g]*weight_mean[not_p,r,g])*MU_phis[g][not_p,r,:])
                    alph = expected_SxW(weight_prob[p,r,g],weight_mean[p,r,g],weight_var[p,r,g])/(sig_qs_1[g])
                    SIGMA_phis[g][p,r,:,:] = update_SIGMA_phi(K,alph,S_g[g])
                    MU_phis[g][p,r,:] = update_MU_phi(SIGMA_phis[g][p,r,:,:],(weight_prob[p,r,g]*weight_mean[p,r,g])/sig_qs_1[g],Y,sum_other)                   
            
             ### update group level edges          
            B_g_mean[g] = np.tile(Fi_mean[:,g,np.newaxis],(1,S_g[g]))
            B_g_variance[g] = np.zeros((coefs,S_g[g],S_g[g]))
            for r in rand_order_coefs:
                for p in rand_order_covariates:
                    exp_w = weight_mean[p,r,g]*weight_prob[p,r,g]
                    var_w = weight_prob[p,r,g]*weight_var[p,r,g]
                    B_g_mean[g][r,:] += MU_phis[g][p,r,:]*exp_w
                    B_g_variance[g][r,:,:] += exp_w**2 * SIGMA_phis[g][p,r,:,:] + MU_phis[g][p,r,:]**2 * var_w + var_w*SIGMA_phis[g][p,r,:,:]

            ### update edge inclusion probabilities
            for r in rand_order_coefs:
                s = 10000
                delta_s = np.random.binomial(1, gamma_B[r,g], s)
                k_s = np.log(gamma_B[r,g]/(1. - gamma_B[r,g]))
                Fi_s = np.random.normal(Fi_mean[r,g], np.sqrt(Fi_var[r,g]), s)
                # Fi_s = Fi_mean[r,g]
                sigma_s = delta_s*(1/np.random.gamma(a_qs_1[g], 1/b_qs_1[g], s)) + (1-delta_s)*(1/np.random.gamma(a_qs_0[g], 1/b_qs_0[g], s))
                beta_s = np.random.multivariate_normal(q_beta_mean[r,grp_sub_id[g]], np.diag(q_beta_variance[r,grp_sub_id[g]]), s)
                sum_pws = np.zeros((S_g[g], s))
                for p in rand_order_covariates:
                    s_s = np.random.binomial(1, weight_prob[p,r,g], s)
                    w_s = np.random.normal(weight_mean[p,r,g], np.sqrt(weight_var[p,r,g]), s)
                    phi_s = np.random.multivariate_normal(MU_phis[g][p,r,:], SIGMA_phis[g][p,r,:,:], s)
                    sum_pws = sum_pws + phi_s.T*s_s*w_s
                dq = delta_s * np.exp(-k_s)/(1 + np.exp(-k_s)) - (1-delta_s) * 1/(1 + np.exp(-k_s))
                logp = -S_g[g]/2*np.log(2*np.pi) - S_g[g]/2 * np.log(sigma_s) - 1/(2*sigma_s)*np.sum((beta_s - (delta_s * (Fi_s + sum_pws)).T)**2, 1) + delta_s * np.log(pi_B) + (1-delta_s) * np.log(1 - pi_B)
                logq = delta_s * np.log(1/(1 + np.exp(-k_s))) + (1-delta_s) * np.log(1 - 1/(1 + np.exp(-k_s)))
                pd = np.sum(dq*(logp - logq))/s
                k[r,g] = k_s + 0.1/n_k * pd
                gamma_B[r,g] = 1./(1 + np.exp(-k[r,g]))
                
            ### update variance of spike and slab
            a_qs_1[g] = 0.5*S_g[g]*np.sum(gamma_B[:,g],0) + a_sig1
            b_qs_1[g] = 0.5*gamma_B[:,g].dot(np.sum(q_beta_mean[:,grp_sub_id[g]]**2 + q_beta_variance[:,grp_sub_id[g]],1)) + b_sig1
            a_qs_0[g] = 0.5*S_g[g]*np.sum(1-gamma_B[:,g],0) + a_sig0
            b_qs_0[g] = 0.5*(1-gamma_B[:,g]).dot(np.sum(q_beta_mean[:,grp_sub_id[g]]**2 + q_beta_variance[:,grp_sub_id[g]],1)) + b_sig0
            for r in rand_order_coefs:
                b_qs_1[g] = b_qs_1[g] + 0.5*gamma_B[r,g]*np.trace(B_g_mean[g][r,np.newaxis].T.dot(B_g_mean[g][r,np.newaxis])+B_g_variance[g][r,:,:]) - gamma_B[r,g]*np.sum(q_beta_mean[r,grp_sub_id[g]]*B_g_mean[g][r,])
                b_qs_0[g] = b_qs_0[g] + 0.5*(1-gamma_B[r,g])*np.trace(B_g_mean[g][r,np.newaxis].T.dot(B_g_mean[g][r,np.newaxis])+B_g_variance[g][r,:,:]) - (1-gamma_B[r,g])*np.sum(q_beta_mean[r,grp_sub_id[g]]*B_g_mean[g][r,])

            sig_qs_1[g] = b_qs_1[g]/a_qs_1[g]
            sig_qs_0[g] = b_qs_0[g]/a_qs_0[g]
            
            ### update noise term
            q_beta_mean_matrix = np.reshape(q_beta_mean,(R,R*L,S),order = 'F')
            q_beta_variance_matrix = np.reshape(q_beta_variance,(R,R*L,S),order = 'F')
            
            for r in range(R):
                sum0 = 0
                sum1 = 0
                sum2 = 0
                for s in grp_sub_id[g]:
                    sum0 += X_in[1:,r,s].dot(X_in[1:,r,s])
                    V = Us[:,:,s].T.dot(Us[:,:,s])
                    for K in range(R*L):
                        sum1 += q_beta_mean_matrix[r,K,s]*(X_in[1:,r,s].dot(Us[:,K,s]))
                        for K_prime in range(R*L):
                            if K == K_prime:
                                sum2 += (q_beta_mean_matrix[r,K,s]*q_beta_mean_matrix[r,K_prime,s]+q_beta_variance_matrix[r,K,s])*V[K,K_prime]
                            else:
                                sum2 += (q_beta_mean_matrix[r,K,s]*q_beta_mean_matrix[r,K_prime,s])*V[K,K_prime]
                d2 = 0.5*sum0 - sum1 + 0.5*sum2 + 1
                d1 = S_g[g]*(n - L)/2.0 + 2
                Xi[g,r,r] = d2 / d1

        weight_change = np.absolute(weight_prob - weight_prob_copy).max((0,1))
        weight_mean_change = np.absolute(weight_mean - weight_mean_copy).max((0,1))
        weight_var_change = np.absolute(weight_var - weight_var_copy).max((0,1))
        q_beta_change = np.absolute(q_beta_copy - q_beta_mean).max()
        MU_change = [np.absolute(MU_phis[i] - MU_copy[i]).max() for i in range(G)]
        SIGMA_change = [np.absolute(SIGMA_phis[i] - SIGMA_copy[i]).max() for i in range(G)]
        sig_q1_change = np.absolute(sig_qs_1 - sig_1_copy).max((0))
        sig_q0_change = np.absolute(sig_qs_0 - sig_0_copy).max((0))
        k_change = np.absolute(k - k_copy).max((0))
        gamma_change = np.absolute(gamma_B - gamma_B_copy).max((0))
        Fi_change = np.absolute(Fi_mean - Fi_copy).max((0))
        
        max_changes = np.array([weight_change.max(),weight_mean_change.max(),weight_var_change.max(),q_beta_change.max(),np.max(MU_change),np.max(SIGMA_change), sig_q1_change.max(), sig_q0_change.max(), gamma_change.max(), Fi_change.max()])# k_change.max()])
        print("finished iteration ",iters,max_changes.max())
        max_change = max_changes.max()

    run_time = time.time() - start_time

    gamma_B_reshape = np.reshape(gamma_B,(R,R,G),order = "F")#  > 0.5
    return(q_beta_mean,weight_prob,weight_mean,weight_var,MU_phis,SIGMA_phis,gamma_B_reshape,B_g_mean,B_g_variance,Fi_mean,Xi)
