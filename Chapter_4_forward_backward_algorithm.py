import pandas as pd
import numpy as np

def fwdFcn(observations, states, start_prob, trans_prob, emm_prob, end_st):    
    """forward part of the algorithm"""
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]*trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * trans_prob[k][end_st] for k in states)
    return fwd, p_fwd

def bkwFcn(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """backward part of the algorithm"""
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations+(None,))):  
        b_curr = {}
        for st in states:
            if i == 0:
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][observation_i_plus] * b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)
    return bkw, p_bkw

def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """forward-backward algorithm"""
    fwd, p_fwd = fwdFcn(observations, states, start_prob, trans_prob, emm_prob, end_st)
    bkw, p_bkw = bkwFcn(observations, states, start_prob, trans_prob, emm_prob, end_st)

    fwd.insert(0,start_prob) 
    
    # combine the forward and backward parts to the state probabilities conditioal on all observations 
    posterior = []
    for i in range(len(observations)+1): 
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    return fwd, bkw, posterior
