import pandas as pd
import numpy as np
import copy 

### FUNCTION DEFINITIONS ####

# Loss Function
def entropy(p_vec, pseudo=0.01):
    if np.sum(p_vec) > 0:
        return np.sum([-(p)*np.log((p)) for p in [(x/np.sum(p_vec))+pseudo for x in p_vec]])
    else:
        return 0

def gini(p_vec):
    if np.sum(p_vec) != 0:
        return (1 - np.sum([(x/np.sum(p_vec))**2 for x in p_vec]))
    else:
        return .5 ## this is not correct, but in the end, it shouldn't matter bc it gets weighted by 0. BE CAREFUL IN FUTURE USE

def two_class_weighted_entropy(counts, pseudo=.01):
    return (entropy([counts[0], counts[1]], pseudo=pseudo)*np.sum(counts[0:2]) + entropy([counts[2], counts[3]], pseudo=pseudo)*np.sum(counts[2:4]))/np.sum(counts)

def two_class_weighted_gini(counts):
    return (gini([counts[0], counts[1]])*np.sum(counts[0:2]) + gini([counts[2], counts[3]])*np.sum(counts[2:4]))/np.sum(counts)


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


# function that creates random initial beta
def random_beta(motif_length):
    output = []
    for i in range(motif_length):
        temp = np.zeros(4)
        num = np.random.choice(range(4))
        temp[num] = np.random.normal(loc=1/(motif_length-1), scale=0.1)
        output.extend(temp)

    return output






### CLASS DEFINITIONS ###

class Node:
    
    def __init__(self, motif_length, seq_length, beta0):
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.thresh = 1
        self.beta = beta0
        self.loss_func = two_class_weighted_gini
        self.terminal = False

    def set_terminal_status(self, status):
        self.terminal = status

    def find_optimal_beta(self, X, y, iterations):
        print("starting beta...", self.beta)
        
        labels = y
        classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, self.beta, self.seq_length, self.motif_length, self.thresh), axis=1)
        #current_entropy = two_class_weighted_entropy(return_counts(labels, classification))
        current_entropy = self.loss_func(return_counts(labels, classification))

        for i in range(iterations):
            print(i)
            print("current entropy...:", current_entropy)
            print("current counts...:", return_counts(labels, classification))
            
            #try new beta
            print('trying new beta')
            new_beta = small_change(self.beta, std=np.random.random())
            new_classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, new_beta, self.seq_length, self.motif_length, self.thresh), axis=1)
            print('counts of new beta....:', return_counts(labels, new_classification))
            
            if self.loss_func(return_counts(labels, new_classification)) < current_entropy:
                print("\nTHIS BETA WAS BETTER!!\n")
                self.beta = new_beta
                classification = new_classification
                current_entropy = self.loss_func(return_counts(labels, new_classification))
            else:
                print("\nback to old beta \n")

        return self.beta


    def split_points(self, X, y):
       
        classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, self.beta, self.seq_length, self.motif_length, self.thresh), axis=1)

        left_split = pd.DataFrame(X).ix[classification == 1]
        left_split_labels = y.ix[classification == 1]

        right_split = pd.DataFrame(X).ix[classification == 0]
        right_split_labels = y.ix[classification == 0]

        return (left_split, left_split_labels), (right_split, right_split_labels)



class ObliqueConvDecisionTree:
    
    def __init__(self, depth, motif_length, seq_length):
        self.depth = depth
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.nodes = []

    def fit(self, X, y, iterations):
        data = []
        artificial_initial_beta = [0,1,0,0,
                                   1,0,0,0,
                                   0,1,0,0,
                                   0,0,1,0,
                                   0,0,0,1,
                                   0,0,1,0]
        artificial_initial_beta = [x/5 for x in artificial_initial_beta]

        for layer in range(self.depth):
            if layer == 0:
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)
                node0.find_optimal_beta(X, y, iterations)
                self.nodes.append([node0])
                data.append([node0.split_points(X, y)])
            else:
                for i in range(len(self.nodes[layer-1])):

                    left_X, left_y = data[layer-1][i][0]
                    right_X, right_y = data[layer-1][i][1]

                    temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                    temp_node_L.find_optimal_beta(left_X, left_y, iterations)
                    
                    temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                    temp_node_R.find_optimal_beta(right_X, right_y, iterations)

                    if i==0:
                        self.nodes.append([temp_node_L, temp_node_R])
                        data.append([temp_node_L.split_points(left_X, left_y), temp_node_R.split_points(right_X, right_y)])
                    else:
                        self.nodes[layer].extend([temp_node_L, temp_node_R])
                        data[layer].extend([temp_node_L.split_points(left_X, left_y), temp_node_R.split_points(right_X, right_y)])













                




    

