import pandas as pd
import numpy as np

# Loss Function
def entropy(p_vec, pseudo=0.01):
    if np.sum(p_vec) > 0:
        return np.sum([-(p)*np.log((p)) for p in [(x/np.sum(p_vec))+pseudo for x in p_vec]])
    else:
        return 0

def two_class_weighted_entropy(counts, pseudo=.01):
    return entropy([counts[0], counts[1]], pseudo=pseudo)*np.sum(counts[0:2]) + entropy([counts[2], counts[3]], pseudo=pseudo)*np.sum(counts[2:4])


# Vector threshold function
def vec_threshold(v, threshold):
    if any(x > threshold for x in v):
        return 1
    else:
        return 0


# Scan across sequence
def scan_convolution(dum_sequence, Beta, seq_length, conv_length):
    return [np.dot(np.array(dum_sequence)[4*i:(4*(i+conv_length))], Beta) for i in range(seq_length-conv_length)]

def classify_sequence(dum_sequence, Beta, seq_length, conv_length, threshold):
    return vec_threshold(scan_convolution(dum_sequence, Beta, seq_length, conv_length), threshold)



#Beta Updates
def random_change(Beta, std=0.5):
    return [b+np.random.normal(scale=std) for b in Beta]

def small_change(Beta, std=0.2):
    length = int(len(Beta)/4)
    random_base = np.random.choice(range(length))
    new_Beta = copy.deepcopy(Beta)
    new_Beta[random_base*4:random_base*4+4] = [b + np.random.normal(scale=std) for b in new_Beta[random_base*4:random_base*4+4]]
    return new_Beta


# returns how many true pos, false, pos, true neg, false neg
def return_counts(DF):
    true1 = len(DF[(DF.label == 1) & (DF.classification == 1)])
    false1 = len(DF[(DF.label == 0) & (DF.classification == 1)])
    true0 = len(DF[(DF.label == 0) & (DF.classification == 0)])
    false0 = len(DF[(df.label == 1) & (DF.classification == 0)])
    
    return [true1, false1, true0, false0]

def return_counts(labels, classifications):
    zipped = list(zip(labels, classifications))
    true1 = zipped.count((1,1))
    false1 = zipped.count((0,1))
    true0 = zipped.count((0,0))
    false0 = zipped.count((1,0))
    return [true1, false1, true0, false0]



