import numpy as np

def vitervi(transition):
    tosses = "251326344212463366565535614566523665561326345621443235213263461435421"
    prob_loaded = {"1": 1/12, "2": 1/12, "3": 1/12, "4": 1/12, "5":1/3, "6":1/3} 
    prob_fair = {"1":1/6, "2":1/6, "3":1/6, "4":1/6, "5":1/6, "6":1/6}
    score_loaded = np.ones(len(tosses))
    score_fair = np.ones(len(tosses))
    score_loaded[0] = 0.5 * prob_loaded[tosses[0]]
    score_fair[0] = 0.5 * prob_fair[tosses[0]]
    
    for i in range(1,len(tosses)):
        score_loaded[i] = max(score_loaded[i-1] * transition[1,1], score_fair[i-1] * transition[1,0]) * prob_loaded[tosses[i]]
        score_fair[i] = max(score_loaded[i-1] * transition[0,1], score_fair[i-1] * transition[0,0]) * prob_fair[tosses[i]]

    state_estimation = np.ones(len(tosses))
    for n, j in enumerate(score_loaded - score_fair):
        if j < 0:
            state_estimation[n] = 0
            
    return(state_estimation)

transition1 = np.array([[0.8, 0.3], [0.2, 0.7]])
transition2 = np.array([[0.9, 0.15], [0.1, 0.85]])
transition3 = np.array([[0.95, 0.05], [0.6, 0.4]])
transition4 = np.array([[0.5, 0.5], [0.5, 0.5]])

vitervi(transition1)
vitervi(transition2)
vitervi(transition3)
vitervi(transition4)