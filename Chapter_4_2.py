################################################

states = ('Healthy', 'Fever', 'Other')
end_state = 'E'
 
observations = ('normal', 'cold', 'dizzy', 'dizzy', 'mad', 'normal')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.3, 'Other': 0.1}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.2, 'Other': 0.1, 'E': 0.01},
   'Fever' : {'Healthy': 0.3, 'Fever': 0.59, 'Other': 0.1, 'E': 0.01},
   'Other' : {'Healthy': 0.3, 'Fever': 0.29, 'Other': 0.4, 'E': 0.01},
}
 
emission_probability = {
   'Healthy' : {'normal': 0.49, 'cold': 0.4, 'dizzy': 0.1, 'mad':.01},
   'Fever' : {'normal': 0.09, 'cold': 0.3, 'dizzy': 0.6, 'mad':.01},
   'Other' : {'normal': 0.09, 'cold': 0.8, 'dizzy': 0.1, 'mad':.01},
}

##############################################################################

# get probabilities of each state conditional on the observations

from Chapter_4_forward_backward_algorithm import *
from Chapter_4_Viterbi import *

# from IPython.display import display
import pprint
import numpy as np

fwd, bkw, state_prob = \
    fwd_bkw(observations, states, start_probability, transition_probability, emission_probability, end_state)

def computeProbs(f):
    periodSum = sum([f[k] for k in f.keys()])
    return [f[k] / periodSum for k in f.keys()]

print(observations)
print('forward')
pd.DataFrame([computeProbs(f) for f in fwd], columns = fwd[0].keys()).transpose()
print(('backward'))
pd.DataFrame([computeProbs(f) for f in bkw], columns = bkw[0].keys()).transpose()
print('state probability')
pd.DataFrame([computeProbs(f) for f in state_prob], columns = state_prob[0].keys()).transpose()

#################################################################################

# determine most likely sequence of events consistent with the obsevations

start_probability = state_prob[0] 
state_prob1 = state_prob[1:] 
V, opt_states,max_prob = viterbi_states(states, start_probability, state_prob1, transition_probability)
print ('The most probable series of states was [{}] with probability {:0.4}\n'.format(', '.join(opt_states), max_prob))

print("observations")
pd.DataFrame([observations, opt_states],index=['observations','optimal states'])
print("probabilities")
pd.DataFrame([{x[key]['prob'] for key in x.keys()} for x in V], columns = V[0].keys()).transpose()
print("prior states")
pd.DataFrame([[x[key]['prior'] for key in x.keys()] for x in V], columns = V[0].keys()).transpose()
