import numpy as np

def getPath(V):
    """Backtrack through lattice to retrieve path to optimal solution"""
    opt_states = []
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt_states.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt_states.insert(0, V[t + 1][previous]["prior"])
        previous = V[t + 1][previous]["prior"]
    return opt_states, max_prob

def viterbi_states(states, start_p, state_c_prob, trans_p):
    """Construct the lattice using the forward algorithm"""
    V = [{}]
    # Initialize the lattice
    for st in states:
        V[0][st] = {"prob": start_p[st], "prior": None}
    # Run Viterbi when t > 0 to incrementally create the lattice
    for t in range(1, len(state_c_prob)):
        V.append({})
        for st in states:
            prev_st_selected = states[0]
            max_prob = V[t-1][states[0]]["prob"] * trans_p[states[0]][st] 
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"] * trans_p[prev_st][st] 
                if tr_prob > max_prob:
                    max_prob = tr_prob
                    prev_st_selected = prev_st
            
            max_prob *= state_c_prob[t][st]        
            V[t][st] = {"prob": max_prob, "prior": prev_st_selected}
                    
    opt_states, max_prob = getPath(V)
    return V, opt_states, max_prob
