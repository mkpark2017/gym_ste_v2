import numpy as np
class queue:
    def __init__(self):
        self.q = []

    def clear(self):
        self.q = []
        
    def append(self, ob):
        self.q.append(ob) #add obs to next column

    def getObservation(self, window_length, ob, pic=False):
        state = ob
        for i in range(window_length):
            if i == 0: continue
            if(i < len(self.q)):
                # load each obs from self.q column and make it single vector
                # if overflow the length, delete first element.
                # if not enough obs, just return as long as possible
                state = np.concatenate((self.q[len(self.q) - i - 1], state))
            else :
                state = np.concatenate((state, ob))
        if pic:
            return np.array(state)
        return np.array(state).ravel()
