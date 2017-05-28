import pandas as pd
import numpy as np
import copy 
from numba import jit, vectorize
import itertools
from numpy.lib.stride_tricks import as_strided
import nltk


### FUNCTION DEFINITIONS ####

# Loss Function
def entropy(p_vec, pseudo=0.00001):
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



# CLASSIFY SEQUENCES
def classify_sequences(X, beta, motif_length, sequence_length):

    X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]

    return threshold(single_sigmoid_vectorized(sig_sum, 100, 0.9))

def newclassify_sequences(X_matrices, beta, motif_length, sequence_length):

    #X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]

    return threshold(single_sigmoid_vectorized(sig_sum, 100, 0.9))

def classify_sequence(x, beta, motif_length, sequence_length):

    x_matrix = x_to_matrix(x, motif_length, sequence_length)
    a = np.dot(x_matrix, beta)
    sig_sum = np.sum(single_sigmoid_vectorized(a, 100, 0.9))

    return threshold(single_sigmoid_vectorized(sig_sum, 100, 0.9))




@vectorize('float64(float64)')
def threshold(value):
    if value > 0.5:
        return 1
    else:
        return 0


###################
##Beta proposals ##
###################
def random_change(Beta, std=0.5):
    return [b+np.random.normal(scale=std) for b in Beta]

def small_change(Beta, std=0.2):
    length = int(len(Beta)/4)
    random_base = np.random.choice(range(length))
    new_Beta = copy.deepcopy(Beta)
    new_Beta[random_base*4:random_base*4+4] = [b + np.random.normal(scale=std) for b in new_Beta[random_base*4:random_base*4+4]]
    return new_Beta

#def medium_change(Beta):
#    length = int(len(Beta)/4)
#    for base in range(length):
#        new_Beta = copy.deepcopy(Beta)
        

def acceptable_beta(Beta, thresh):
    stride = Beta[0].itemsize
    Beta_ndarray = as_strided(Beta, shape=(6,4), strides=[stride*4,stride])
    return all(np.sum(np.abs(Beta_ndarray), axis=1) < thresh)



# returns how many true pos, false, pos, true neg, false neg
def return_counts(labels, classifications):
    zipped = list(zip(labels, classifications))
    true1 = zipped.count((1,1))
    false1 = zipped.count((0,1))
    true0 = zipped.count((0,0))
    false0 = zipped.count((1,0))
    return [true1, false1, true0, false0]

def return_weightedcounts(labels, classifications, weights):
    zipped = list(zip(list(zip(labels, classifications)), weights))
    true1 = np.sum([a[1] for a in zipped if a[0]==(1,1)])
    false1 = np.sum([a[1] for a in zipped if a[0]==(0,1)]) 
    true0 = np.sum([a[1] for a in zipped if a[0]==(0,0)])
    false0 = np.sum([a[1] for a in zipped if a[0]==(1,0)])
    return [true1, false1, true0, false0]




# function that creates random initial beta
def random_beta(motif_length):
    output = []
    for i in range(motif_length):
        temp = np.zeros(4)
        num = np.random.choice(range(4))
        temp[num] = np.random.normal(loc=1/(motif_length-1), scale=0.01)
        output.extend(temp)

    return output

#######################################
##### GRADIENT DESCENT FUNCTIONS ######
#######################################
def single_sigmoid(x, alpha=100, offset=0.9):
    return 1/(1 + np.exp(-alpha*(x-offset)))

@vectorize('float64(float64, float64, float64)')
def single_sigmoid_vectorized(x, alpha=100, offset=0.9):
    return 1/(1 + np.exp(-alpha*(x-offset)))

@vectorize('float64(float64)')
def simplified_sigmoid(x):
    return 1/(1 + x)

def single_sigmoid_deriv(x, alpha=100, offset=0.9):
    exponent = np.exp(-alpha*(x-offset))
    return (alpha * exponent) / (1 + exponent)**2

@vectorize('float64(float64, float64, float64)')
def single_sigmoid_deriv_vectorized(x, alpha=100, offset=0.9):
    exponent = np.exp(-alpha*(x-offset))
    return (alpha * exponent) / (1 + exponent)**2

@vectorize('float64(float64, float64)')
def simplified_sigmoid_deriv(x, alpha):
    return (alpha * x)/((1 + x)**2)



def x_to_string(x):
    return "".join([str(i) for i in x])


def x_to_matrix(x, motif_length, sequence_length):
    numpy_arrayx = np.array(x)
    size = numpy_arrayx.itemsize

    #print('size', size)
    return as_strided(numpy_arrayx, shape = [sequence_length - motif_length, motif_length*4], strides = [size*4,size])


### returns the sum of all the sigmoids of all subsequences
def sum_sigmoid_sequence(xdotbeta, motif_length, sequence_length):

    #x_matrix = x_to_matrix(x, motif_length, sequence_length)
    vectorized_single_sigmoid = np.vectorize(single_sigmoid)

    return np.sum(vectorized_single_sigmoid(xdotbeta, alpha=100, offset=0.9))

@jit
def better_sum_sigmoid_sequence(x, beta, motif_length, sequence_length):
    x_matrix = x_to_matrix(x, motif_length, sequence_length)

    output = 0
    for m in np.dot(x_matrix, beta):
        output += m

    return output



def sum_sigmoid_deriv_sequence(x, beta, motif_length, sequence_length):

    x_matrix = x_to_matrix(x, motif_length, sequence_length)
    vectorized_single_sigmoid_deriv = np.vectorize(single_sigmoid_deriv)    

    return np.sum(np.dot(np.diag(vectorized_single_sigmoid_deriv(np.dot(x_matrix, beta), alpha=100, offset=0.9)), x_matrix), 
            axis=0)



    def gradient(X, y, beta, motif_length, sequence_length):

        X_positive = X[y==1]
    X_negative = X[y==0]


    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    T = [0, 0, 0, 1]
    nucleotides = [A, C, G, T]

    combinations = [[item for sublist in p for item in sublist] for p in itertools.product(nucleotides, repeat=6)]
    combinations = ["".join([str(x) for x in combo]) for combo in combinations]

    combination_lookupvalues = np.exp(-100 * (np.dot(combinations, beta) - 0.9))

    lookuptable = dict(zip(combinations, combination_lookupvalues))


    total = []

    p = np.sum([single_sigmoid(sum_sigmoid_sequence(lookuptable[x_to_string(X.ix[i])], motif_length, sequence_length)) for i in y[y==1].index.values])
    n = np.sum([single_sigmoid(sum_sigmoid_sequence(lookuptable[x_to_string(X.ix[i])], beta, motif_length, sequence_length)) for i in y[y==0].index.values])

    P = len(y[y==1])
    N = len(y[y==0])

    print(p, n, P-p, N-n)

    p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))


    for x in np.array(X):
        S = sum_sigmoid_sequence(x, beta, motif_length, sequence_length)
        first_term = single_sigmoid_deriv(S)
        second_term = sum_sigmoid_deriv_sequence(x, beta, motif_length, sequence_length)

        total.append(first_term * second_term)

    first = entropy([P, N])
    second = two_class_weighted_entropy([p, n, N-n, P-p])

    output = np.dot(np.array(p_factor), np.array(total))
    return output/(np.sum(np.abs(output))*8) , (first - second)

def newnewgradient(X_matrices, y, beta, motif_length, sequence_length, step_size=1/50):

    #X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]
    b = [single_sigmoid_deriv_vectorized(x, 100, 0.9) for x in a]
    c = [np.sum(X_matrices[i] * b[i][:,np.newaxis], axis=0) for i in range(len(X_matrices))]
    d = [single_sigmoid_deriv(x) for x in sig_sum]

    p = pd.Series(sig_sum)[(y==1).as_matrix()].apply(single_sigmoid, args=(100,0.9)).sum()
    n = pd.Series(sig_sum)[(y==0).as_matrix()].apply(single_sigmoid, args=(100,0.9)).sum()

    P = len(y[y==1])
    N = len(y[y==0])

    p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))

    gradient = np.sum((c * np.array(d)[:, np.newaxis]) * p_factor[:, np.newaxis], axis=0)

    return (gradient/(np.sqrt(np.dot(gradient,gradient)) * (1/step_size))), [p, n, N-n, P-p]

def weightedgradient(X_matrices, y, weights, beta, motif_length, sequence_length, step_size=1/50):
    weights_series = pd.Series(weights)

    #X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
    a = np.array([np.dot(x, beta) for x in X_matrices])
    sig_sum = [np.sum(single_sigmoid_vectorized(x, 100, 0.9)) for x in a]
    b = [single_sigmoid_deriv_vectorized(x, 100, 0.9) for x in a]
    c = [np.sum(X_matrices[i] * b[i][:,np.newaxis], axis=0) for i in range(len(X_matrices))]
    d = [single_sigmoid_deriv(x) for x in sig_sum] * weights
    #print(len(d))

    p = (pd.Series(sig_sum)[(y==1)].apply(single_sigmoid, args=(100,0.9)) * weights[y==1]).sum()
    n = (pd.Series(sig_sum)[(y==0)].apply(single_sigmoid, args=(100,0.9)) * weights[y==0]).sum()

    #print(p, n)

    P = weights_series[y==1].sum()
    N = weights_series[y==0].sum()

    #p_factor = (y==0).apply(lambda x: int(x))*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1).apply(lambda x: int(x))*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))
    p_factor = (y==0)*(np.log(n/(N-n))-np.log((p+n)/(P+N-p-n)))+(y==1)*(np.log(p/(P-p))-np.log((p+n)/(P+N-p-n)))

    gradient = np.sum((c * np.array(d)[:, np.newaxis]) * p_factor[:, np.newaxis], axis=0)

    #return (gradient/np.sum(np.abs(gradient) * 15)), [p, n, N-n, P-p]
    return (gradient/(np.sqrt(np.dot(gradient,gradient)) * (1/step_size))), [p, n, N-n, P-p]


def Information_Gain(X, y, beta, motif_length, sequence_length):

    p = np.sum([single_sigmoid(sum_sigmoid_sequence(X.ix[i], beta, motif_length, sequence_length)) for i in y[y==1].index.values])
    n = np.sum([single_sigmoid(sum_sigmoid_sequence(X.ix[i], beta, motif_length, sequence_length)) for i in y[y==0].index.values])

    P = len(y[y==1])
    N = len(y[y==0])

    first = entropy([P, N])
    second = two_class_weighted_entropy([p, n, P-p, N-n])

    return first - second


def acceptance_probability(initial, final, T):
    return np.exp((initial - final)/T)




#### N GRAM COUNTIN #####
def getSequenceNgrams(sequence):
    return [''.join(x) for x in nltk.ngrams(sequence, 6)]

def getCounts(sequences):
    count_dict = {}
    ngrams = sequences.apply(getSequenceNgrams)
    for i in sequences.index.values:
        for gram in ngrams.ix[i]:
            if gram not in count_dict.keys():
                count_dict[gram] = 1
            else:
                count_dict[gram] += 1
    return count_dict

def motif_to_beta(motif):
    A = [1.0,0.0,0.0,0.0]
    C = [0.0,1.0,0.0,0.0]
    G = [0.0,0.0,1.0,0.0]
    T = [0.0,0.0,0.0,1.0]
    convertdict = {'A':A, 'C':C, 'G':G, 'T':T}

    return np.array([convertdict[x] for x in motif]).flatten()

def normalize_dict(d):
    d_copy = copy.deepcopy(d)
    total = sum(d.values())
    for k in d_copy:
        d_copy[k] /= total
    return d_copy



#########################
### CLASS DEFINITIONS ###
#########################

class Node:

    def __init__(self, motif_length, seq_length, beta0):
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.thresh = 1
        self.beta = beta0
        self.loss_func = two_class_weighted_entropy
        self.terminal = False
        self.left_classification = None
        self.right_classification = None
        self.loss_memory = []

    def set_loss_function(loss):
        self.loss_func = loss

    def set_terminal_status(self, status):
        self.terminal = status


    def fit(self, X_matrices, y, weights, iterations, step_size):
        data_size = len(X_matrices)
        labels = y


        #X_matrices = [x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)]

        for i in range(iterations):
            grad = weightedgradient(X_matrices,  y, weights, self.beta, self.motif_length, self.seq_length, step_size)

            self.beta += grad[0]
            self.loss_memory.append(self.loss_func(grad[1]))


        classification = newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length)
        print("counts...", return_counts(labels, classification))
        current_entropy = self.loss_func(return_counts(labels, classification))
        print("current entropy...", current_entropy)


        ## ONCE FIT, final classification of resulting nodes are defined ##
        final_counts = return_counts(labels, classification)
        if final_counts[0] > final_counts[1]:
            self.left_classification = 1
        else:
            self.left_classification = 0

        if final_counts[2] > final_counts[3]:
            self.right_classification = 0
        else:
            self.right_classification = 1



    def anneal(self, X_matrices, y, weights, alpha=0.9, T_start = .001, T_min = 0.0005, iterT=100):
        #X_matrices = [x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)]
        #cost = self.loss_func(return_weightedcounts(y, newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length), weights))
        cost = self.loss_func(return_weightedcounts(y, newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length), weights))
        T = T_start

        while T > T_min:
            i = 1
            print('New Temperature', T, "\n")
            while i <= iterT:
                ## VERY IMPORTANT STEP!!! POTENTIAL GAINS HERE in better proposals##

                while True:
                    new_beta = small_change(self.beta, std=np.random.chisquare(.5))
                    #new_beta = random_change(self.beta, std=np.random.chisquare(.4))
                    if acceptable_beta(new_beta,thresh=1):
                        break

                #new_cost = self.loss_func(return_weightedcounts(y, newclassify_sequences(X_matrices, new_beta, self.motif_length, self.seq_length), weights))
                new_cost = self.loss_func(return_weightedcounts(y, newclassify_sequences(X_matrices, new_beta, self.motif_length, self.seq_length), weights))
                ap = acceptance_probability(cost, new_cost, T)
                #print(ap)
                if ap > np.random.random():
                    self.beta = new_beta
                    cost = new_cost
                    self.loss_memory.append(cost)
                i += 1
            T *= alpha

### everything here and below copied from fit
        classification = newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length)
        print("counts...", return_counts(y, classification))
        current_entropy = self.loss_func(return_counts(y, classification))
        print("current entropy...", current_entropy)


        ## ONCE FIT, final classification of resulting nodes are defined ##
        final_counts = return_counts(y, classification)
        if final_counts[0] > final_counts[1]:
            self.left_classification = 1
        else:
            self.left_classification = 0

        if final_counts[2] > final_counts[3]:
            self.right_classification = 0
        else:
            self.right_classification = 1








        #return self.beta



    def split_points(self, X_matrices, y, weights):

        classification = newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length)

        left_split = np.array(X_matrices)[classification==1]
        left_split_labels = np.array(y)[classification == 1]
        left_split_weights = weights[classification==1]

        right_split = np.array(X_matrices)[classification==0]
        right_split_labels = np.array(y)[classification == 0]
        right_split_weights = weights[classification==0]

        return (left_split, left_split_labels, left_split_weights), (right_split, right_split_labels, right_split_weights)

    def newsplit_points(self, indices, X_matrices):

        classification = newclassify_sequences(X_matrices, self.beta, self.motif_length, self.seq_length)

        left_split = indices[np.where(classification==1)[0]]
        right_split = indices[np.where(classification==0)[0]]

        return (left_split, right_split)

    def predict_one(self, x):
        if classify_sequence(x, self.beta, self.motif_length, self.seq_length) > 0.5:
            return self.left_classification
        else:
            return self.right_classification

    def predict(self, X):

        return classify_sequences(X, self.beta, self.motif_length, self.seq_length)





class ObliqueConvDecisionTree:

    def __init__(self, depth, motif_length, seq_length, initial_betas, initial_beta_probabilities):
        self.depth = depth
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.initial_betas = initial_betas
        self.initial_beta_probabilities = initial_beta_probabilities
        self.nodes = []

    def gradientfit(self, X, y, weights, iterations, step_size):
        data = []

        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)])

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                #node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                node0.fit(X_matrices, y, weights, iterations, step_size)
                #node0.anneal(X_matrices, y, weights, alpha=0.9, T_start=.0005, T_min=0.0001, iterT=200)

                self.nodes.append([node0])
                data.append([node0.newsplit_points(np.arange(len(X_matrices)), X_matrices)])

            #Rest of the layers
        else:

            #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:

                        left = data[layer-1][i][0]
                        right = data[layer-1][i][1]

                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_L.fit(X_matrices.take(left, axis=0), y.take(left), weights.take(left), iterations, step_size)
                        #temp_node_L.anneal(X_matrices.take(left, axis=0), y.take(left), weights.take(left), alpha=.9, T_start=.0005, T_min=.0001, iterT=200)

                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_R.fit(X_matrices.take(right, axis=0), y.take(right), weights.take(right), iterations, step_size)
                        #temp_node_R.anneal(X_matrices.take(right, axis=0), y.take(right), weights.take(right), alpha=.9, T_start=.0005, T_min=.0001, iterT=200)

                        left_children = temp_node_L.newsplit_points(left, X_matrices.take(left, axis=0))
                        right_children = temp_node_R.newsplit_points(right, X_matrices.take(right, axis=0))


                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        if (np.min([len(left_children[0]), len(left_children[1])]) < .05*len(X_matrices)):
                            temp_node_L.set_terminal_status(status=True)
                        else:
                            pass

                        if (np.min([len(right_children[0]), len(right_children[1])]) < .05*len(X_matrices)):
                            temp_node_R.set_terminal_status(status=True)
                        else:
                            pass

                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([left_children, right_children])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([left_children, right_children])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))

                        temp_node_L.set_terminal_status(status=True)
                        temp_node_R.set_terminal_status(status=True)



                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([data[layer-1][i], data[layer-1][i]])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([data[layer-1][i], data[layer-1][i]])

        for node in  self.nodes[-1]:
            node.set_terminal_status(status=True)



    def annealfit(self, X, y, weights, alpha=0.9, T_start=.0005, T_min=.0001, iterations=250):
        data = []

        X_matrices = np.array([x_to_matrix(x, self.motif_length, self.seq_length) for x in np.array(X)])

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                        beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                node0.anneal(X_matrices, y, weights, alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                self.nodes.append([node0])
                data.append([node0.newsplit_points(np.arange(len(X_matrices)), X_matrices)])

            #Rest of the layers
        else:

            #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:

                        left = data[layer-1][i][0]
                        right = data[layer-1][i][1]

                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_L.anneal(X_matrices.take(left, axis=0), y.take(left), weights.take(left), alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, 
                                beta0=motif_to_beta(np.random.choice(self.initial_betas, p=self.initial_beta_probabilities))/(self.motif_length-1))
                        temp_node_R.anneal(X_matrices.take(right, axis=0), y.take(right), weights.take(right), alpha=alpha, T_start=T_start, T_min=T_min, iterT=iterations)

                        left_children = temp_node_L.newsplit_points(left, X_matrices.take(left, axis=0))
                        right_children = temp_node_R.newsplit_points(right, X_matrices.take(right, axis=0))


                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        if (np.min([len(left_children[0]), len(left_children[1])]) < .05*len(X_matrices)):
                            temp_node_L.set_terminal_status(status=True)
                        else:
                            pass

                        if (np.min([len(right_children[0]), len(right_children[1])]) < .05*len(X_matrices)):
                            temp_node_R.set_terminal_status(status=True)
                        else:
                            pass

                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([left_children, right_children])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([left_children, right_children])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))

                        temp_node_L.set_terminal_status(status=True)
                        temp_node_R.set_terminal_status(status=True)

                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([data[layer-1][i], data[layer-1][i]])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([data[layer-1][i], data[layer-1][i]])

        for node in  self.nodes[-1]:
            node.set_terminal_status(status=True)


    def predict_one(self, x):

        #start with the head node
        current_layer = 0
        leftright = 0
        current_node = self.nodes[current_layer][leftright]

        terminal_node = False
        #loop uniil at terminal node
        while terminal_node == False:

            out = classify_sequence(x, current_node.beta, current_node.motif_length, current_node.seq_length)
            if out == 1:
                current_layer += 1
                leftright = leftright*2
                current_node = self.nodes[current_layer][leftright]

            else:
                current_layer += 1
                leftright = leftright*2 + 1
                current_node = self.nodes[current_layer][leftright]

            terminal_node = current_node.terminal

        return current_node.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])



class BoostedConvDT:

    def __init__(self, num_trees, tree_depth):
        self.num_trees = num_trees
        self.tree_depth = tree_depth


################################################################################################################3


def print_with_features(L, Features, ordered=False):
    if ordered==False:
        for i in range(len(features)):
            print(L[i], Features[i])
    else:
        print_with_features([L[x] for x in np.argsort(L)[::-1]],
                [Features[x] for x in np.argsort(L)[::-1]])

def update_importances(importances, tree, weights, alpha):
    if len(importances) != len(weights):
                raise ValueError
    else:
        for i in range(len(importances)):
            importances[i] += tree.feature_importances_ * weights[i] * alpha

def normalize(x):
    return x/np.sum(x)

def predict_proba_importances(X, BDTLIST):
    output = []
    for b in BDTLIST:
        output.append(b.predict(X.reshape(1,-1))[0])

    return output




def plot_roc(true_y, proba_y):
    plt.figure(figsize=(8,5))
    false_pos, true_pos, _ = roc_curve(true_y, proba_y)
    roc_auc = auc(false_pos, true_pos)

    plt.plot(false_pos, true_pos)
    plt.text(.6,.1,"AUC: " + str("%.4f" % roc_auc), fontsize=20)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")




class AdaboostedDecisionTree():

    def __init__(self, initial_betas, initial_beta_probabilities, num_trees=25, max_depth=2, motif_length=6, sequence_length=199):
        self.weights_list = []
        self.importances_list = []
        self.gammas_list = []
        self.trees_list = []
        self.num_trees = num_trees
        self.weights = []
        self.all_importances = []
        self.depth=max_depth
        self.motif_length=motif_length
        self.seq_length=sequence_length
        self.initial_betas = initial_betas
        self.initial_beta_probabilities = initial_beta_probabilities



    def gradientfit(self, X, y, iterations=1000, step_size=1/200):

        self.weights = np.ones(len(X))/len(X)

        for i in range(self.num_trees):
            print("TREE NUMBER", i)
            self.weights_list.append(self.weights)

            t = ObliqueConvDecisionTree(depth=self.depth, motif_length=self.motif_length, seq_length=self.seq_length,
                    initial_betas=self.initial_betas, initial_beta_probabilities=self.initial_beta_probabilities)
            t.gradientfit(X, y, self.weights, iterations, step_size)

            wrong_list = [int(x) for x in t.predict(np.array(X)) != y]
            err = np.sum(self.weights * wrong_list)/np.sum(self.weights)
            gamma = np.log((1-err)/err)
            self.gammas_list.append(gamma)

            self.weights *= np.exp([gamma*x for x in wrong_list]) / np.sum(np.exp([gamma*x for x in wrong_list]))
            self.weights = normalize(self.weights)

            self.trees_list.append(t)

    def annealfit(self, X, y, alpha=0.9, T_start=.0005, T_min=.0001, iterations=250):

        self.weights = np.ones(len(X))/len(X)

        for i in range(self.num_trees):
            print("TREE NUMBER", i)
            self.weights_list.append(self.weights)

            t = ObliqueConvDecisionTree(depth=self.depth, motif_length=self.motif_length, seq_length=self.seq_length,
                    initial_betas=self.initial_betas, initial_beta_probabilities=self.initial_beta_probabilities)
            t.annealfit(X, y, self.weights, alpha, T_start, T_min, iterations)

            wrong_list = [int(x) for x in t.predict(np.array(X)) != y]
            err = np.sum(self.weights * wrong_list)/np.sum(self.weights)
            gamma = np.log((1-err)/err)
            self.gammas_list.append(gamma)

            self.weights *= np.exp([gamma*x for x in wrong_list]) / np.sum(np.exp([gamma*x for x in wrong_list]))
            self.weights = normalize(self.weights)

            self.trees_list.append(t)



    def predict(self, X):

        tree_predictions = np.array([tree.predict(X) for tree in self.trees_list])

        return threshold(np.dot(self.gammas_list, tree_predictions)/np.sum(self.gammas_list))



class BaggedConvDT():

    def __init__(self, initial_betas, initial_beta_probabilities, num_trees=25, max_depth=2, motif_length=6, sequence_length=199):
        self.trees_list = []
        self.num_trees = num_trees
        self.motif_length = motif_length
        self.sequence_length = sequence_length
        self.depth = max_depth
        self.initial_betas = initial_betas
        self.initial_beta_probabilities = initial_beta_probabilities


    def annealfit(self, X, y, alpha=0.9, T_start=.002, T_min=.0001, iterations=250, percent_bag=0.66):

        self.weights = np.ones(len(X))/len(X)

        for i in range(self.num_trees):
            print("TREE NUMBER", i)

            train_size = int(len(X)*percent_bag)
            rows = np.random.choice(range(X.shape[0]), size=train_size, replace=True)
            X_temp = X[rows]
            y_temp = y[rows]

            t = ObliqueConvDecisionTree(depth=self.depth, motif_length=self.motif_length, seq_length=self.sequence_length,
                    initial_betas=self.initial_betas, initial_beta_probabilities=self.initial_beta_probabilities)
            t.annealfit(X_temp, y_temp, self.weights, alpha, T_start, T_min, iterations)

            self.trees_list.append(t)

    def predict(self, X):
        tree_predictions = np.mean(np.array([tree.predict(X) for tree in self.trees_list]), axis=0)
        return threshold(tree_predictions)






