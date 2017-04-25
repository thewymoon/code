import pandas as pd
import numpy as np
import copy 
from numba import jit, vectorize
import itertools
from numpy.lib.stride_tricks import as_strided


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



# CLASSIFY SEQUENCES
def classify_sequences(X, beta, motif_length, sequence_length):
    X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
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

#def x_to_matrix(x, motif_length, sequence_length):
#    return np.array([list(x[(4*i):(4*(i+motif_length))]) for i in range(sequence_length-motif_length)])

#@jit
#def x_to_matrix(x, motif_length, sequence_length):
#    output = np.array([])
#    for i in range(sequence_length - motif_length):
#        output = np.append(output, x[(4*i):(4*(i+motif_length))])
#
#    return output

def x_to_matrix(x, motif_length, sequence_length):
    numpy_arrayx = np.array(x)
    size = numpy_arrayx.itemsize

    #print('size', size)
    return as_strided(numpy_arrayx, shape = [sequence_length - motif_length, motif_length*4], strides = [size*4,size])


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

def newnewgradient(X, y, beta, motif_length, sequence_length, step_size=1/50):
    #print(len(X))
    #print(len(y))

    X_matrices = [x_to_matrix(x, motif_length, sequence_length) for x in np.array(X)]
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

    def randomfit(self, X, y, iterations):
        data_size = len(X)

        print("starting beta...", self.beta)
        
        labels = y
        classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, self.beta, self.seq_length, self.motif_length, self.thresh), axis=1)
        #current_entropy = two_class_weighted_entropy(return_counts(labels, classification))
        current_entropy = self.loss_func(return_counts(labels, classification))

        for i in range(iterations):
            print(i)
            print("current entropy...:", current_entropy)
            print("current counts...:", return_counts(labels, classification))

            self.loss_memory.append(current_entropy)
            
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



        #return self.beta

    
    def fit(self, X, y, iterations, step_size):
        data_size = len(X)

        #print("starting beta...", self.beta)
        
        labels = y
        #classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, self.beta, self.seq_length, self.motif_length, self.thresh), axis=1)
        #print("counts...", return_counts(labels, classification))
        #current_entropy = self.loss_func(return_counts(labels, classification))

        #print("initial information gain", Information_Gain(X, y, self.beta, self.motif_length, self.seq_length))

        
        #print("current entropy...", current_entropy)
        

        for i in range(iterations):
            #print(i)
            #print('trying new beta')
            grad = newnewgradient(X, y, self.beta, self.motif_length, self.seq_length, step_size)
            #print(grad[0])
            self.beta += grad[0]
            self.loss_memory.append(self.loss_func(grad[1]))
            #print(grad[1])

                
        #classification = pd.DataFrame(X).apply(lambda x: classify_sequence(x, self.beta, self.seq_length, self.motif_length, self.thresh), axis=1)
        classification = classify_sequences(X, self.beta, self.motif_length, self.seq_length)
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



        #return self.beta

    def split_points(self, X, y):
       
        classification = classify_sequences(X, self.beta, self.motif_length, self.seq_length)

        left_split = pd.DataFrame(X).ix[classification == 1]
        left_split_labels = y.ix[classification == 1]

        right_split = pd.DataFrame(X).ix[classification == 0]
        right_split_labels = y.ix[classification == 0]

        return (left_split, left_split_labels), (right_split, right_split_labels)

    def predict_one(self, x):
        if classify_sequence(x, self.beta, self.motif_length, self.seq_length) > 0.5:
            return self.left_classification
        else:
            return self.right_classification

    def predict(self, X):

        #return pd.DataFrame(X).apply(lambda x: self.predict_one(x), axis=1)
        return classify_sequences(X, self.beta, self.motif_length, self.seq_length)





class ObliqueConvDecisionTree:
    
    def __init__(self, depth, motif_length, seq_length):
        self.depth = depth
        self.motif_length = motif_length
        self.seq_length = seq_length
        self.nodes = []

    def fit(self, X, y, iterations, step_size):
        data = []
        artificial_initial_beta = [0,1,0,0,
                                   1,0,0,0,
                                   0,1,0,0,
                                   0,0,1,0,
                                   0,0,0,1,
                                   0,0,1,0]
        artificial_initial_beta = [x/5 for x in artificial_initial_beta]

        for layer in range(self.depth):
            #First layer go!
            if layer == 0:
                #node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)
                node0 = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                node0.fit(X, y, iterations, step_size)
                self.nodes.append([node0])
                data.append([node0.split_points(X, y)])

            #Rest of the layers
            else:

                #loop through the nodes from previous layer
                for i in range(len(self.nodes[layer-1])):

                    ### do this stuff only if the node was not terminal ###
                    if self.nodes[layer-1][i].terminal == False:
                    
                        left_X, left_y = data[layer-1][i][0]
                        right_X, right_y = data[layer-1][i][1]

                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        #temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)
                        temp_node_L.fit(left_X, left_y, iterations, step_size)
                        
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=random_beta(self.motif_length))
                        #temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)
                        temp_node_R.fit(right_X, right_y, iterations, step_size)

                        ######################################################################
                        #### Call it a terminal node if the child nodes don't have enough ####
                        ######################################################################
                        if (np.min([len(temp_node_L.split_points(left_X, left_y)[0][0]), len(temp_node_L.split_points(left_X, left_y)[1][0])]) < .05*len(X)):
                            temp_node_L.set_terminal_status(status=True)
                        else:
                            pass

                        if (np.min([len(temp_node_R.split_points(right_X, right_y)[0][0]), len(temp_node_R.split_points(right_X, right_y)[1][0])]) < .05*len(X)):
                            temp_node_R.set_terminal_status(status=True)
                        else:
                            pass
                        
                        ######################################
                        ### Add the nodes and data to list ###
                        ######################################
                        if i==0:
                            self.nodes.append([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data.append([temp_node_L.split_points(left_X, left_y), temp_node_R.split_points(right_X, right_y)])
                        else:
                            self.nodes[layer].extend([copy.deepcopy(temp_node_L), copy.deepcopy(temp_node_R)])
                            data[layer].extend([temp_node_L.split_points(left_X, left_y), temp_node_R.split_points(right_X, right_y)])


                    else: #make dummy nodes and set status to terminal also
                        temp_node_L = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)
                        temp_node_R = Node(motif_length=self.motif_length, seq_length=self.seq_length, beta0=artificial_initial_beta)

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
            #print(current_layer)
            #print(leftright)    

            out = classify_sequence(x, current_node.beta, current_node.motif_length, current_node.seq_length)
            if out == 1:
                current_layer += 1
                leftright = leftright*2
                current_node = self.nodes[current_layer][leftright]
                #print('went left')
                
            else:
                current_layer += 1
                leftright = leftright*2 + 1
                current_node = self.nodes[current_layer][leftright]
                #print('went right')

            terminal_node = current_node.terminal

        #print(current_layer, leftright)

        return current_node.predict_one(x)















                




    

