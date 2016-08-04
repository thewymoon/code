import numpy as np
import pandas as pd

################################
# specific to MITF-Myc project #
################################

def OnevAll_SplitTrainTest(df, signal, TF_list):
    if len(df[df.which==signal]) > df.which.value_counts().min()*3:
        training_num = df.which.value_counts().min()*2
        test_num = df.which.value_counts().min()
    else:
        training_num = int(len(df[df.which==signal])*.666)
        test_num = len(df[df.which==signal]) - training_num
        
    train_rows_signal = np.random.choice(df[df.which==signal].index.values, training_num, replace=False)
    train_rows_background = []
    for i in [x for x in TF_list if x != signal]:
        train_rows_background = np.append(train_rows_background,
                                          np.random.choice(df[df.which==i].index.values, training_num*.333, replace=False))
    
    test_rows_signal = np.random.choice(df[df.which==signal].drop((train_rows_signal)).index.values, test_num, replace=False)
    test_rows_background = []
    for i in [x for x in TF_list if x != signal]:
        test_rows_background = np.append(test_rows_background, 
                                        np.random.choice(df.drop((train_rows_background))[df.drop((train_rows_background)).which==i].index.values,
                                        test_num*.333, replace=False))
    df_train = df.ix[np.hstack((train_rows_signal, train_rows_background))]
    df_test = df.ix[np.hstack((test_rows_signal, test_rows_background))]
    
    return df_train, df_test

def convert_to_binary(x, motif):
    if x == motif:
        return 1
    else:
        return 0


#imports data from coverage narrowPeak files and returns dataframe with coverage values for each site
def import_coverage_data(directory, coverage_directory, filename, column_names):
    from os import listdir

    DF = pd.read_table(directory+filename, names=column_names)

    for fname in listdir(coverage_directory):
        name = fname.split(".narrowPeak")[0]
        mylist = []
        for line in open(coverage_directory+fname):
            Columns = line.split("\t")
            mylist.append(int(Columns[7]))
        DF[name.replace('.','_')] = mylist

    return DF


def make_test_train(DF, column_name, train_percent):
    min_class = DF[column_name].value_counts().idxmin()
    classes = DF[column_name].unique()

    train_rows = []
    test_rows = []

    training_num = int(len(DF[DF[column_name]==min_class])*train_percent)
    test_num = len(DF[DF[column_name]==min_class]) - training_num

    for c in classes:
        temp_df = DF[DF[column_name]==c].sample(training_num + test_num)
        rows = np.random.choice(temp_df.index.values, training_num, replace=False)
        train_rows.extend(rows)
        test_rows.extend(temp_df.drop(rows).index.values)


    return DF.ix[train_rows], DF.ix[test_rows]





'''
partition dataframe into train and test set using the most common class to determine size of train
and test set and upsampling other classes to create equal number of each class in final sets
'''
def make_test_train_MaxUP(DF, column_name, train_percent):
    from sklearn.utils import resample

    max_class = DF[column_name].value_counts().idxmax()
    classes = DF[column_name].unique()

    train_rows = []
    test_rows = []

    training_num = int(len(DF[DF[column_name]==max_class])*train_percent)
    test_num = len(DF[DF[column_name]==max_class]) - training_num

    for c in classes:
        if c==max_class:
            temp_df = DF[DF[column_name]==c]
            rows = np.random.choice(temp_df.index.values, test_num, replace=False)
            test_rows.extend(rows)
            train_rows.extend(temp_df.drop(rows).index.values)
        else:
            rows = DF[DF[column_name]==c].sample(test_num).index.values
            test_rows.extend(rows)
            blah = np.random.choice(DF[DF[column_name]==c].drop(rows).index.values, training_num, replace=True)
            train_rows.extend(blah)

    return DF.ix[train_rows], DF.ix[test_rows]


        
    



###########################################
# scikit-learn and BDT-specific functions #
###########################################

def get_importances_dict(bdt, branch_names):
    importances = bdt.feature_importances_
    indices = np.argsort(importances)[::-1]
    importances_dict = {}
    for i in range(len(branch_names)):
        importances_dict[branch_names[indices[i]]] = i
    return importances_dict

def order_by_importance(bdt, branch_names):
    importances = bdt.feature_importances_
    indices = np.argsort(importances)[::-1]
    importances_list = []
    for i in range(len(branch_names)):
        importances_list.append(branch_names[indices[i]])
    return importances_list

def find_depths_in_tree(feature_index, tree):
    depths_in_tree = []
    left_children = list(tree.tree_.children_left)
    right_children = list(tree.tree_.children_right)

    for i in np.where(tree.tree_.feature == feature_index)[0]:
        index = i
        count = 0
        while index != 0:
            if index in left_children:
                index = left_children.index(index)
                count += 1
                #print "left"
            elif index in right_children:
                index = right_children.index(index)
                count += 1
                #print "right"
            else:
                print "something's wrong..."
                index = 0

        #print count
        depths_in_tree.append(count)
    return depths_in_tree

def find_all_depths(feature_index, BDT):
    all_depths = []
    for tree in BDT.estimators_:
        all_depths += find_depths_in_tree(feature_index, tree)
    return all_depths

def print_importances(C, feature_names):
    Importances = C.feature_importances_
    Indices = np.argsort(Importances)[::-1]
    for i in range(len(feature_names)):
        print feature_names[Indices[i]] + ':', str(Importances[Indices[i]])

def return_most_important_features(C, feature_names, num):
    Importances = C.feature_importances_
    Indices = np.argsort(Importances)[::-1]

    return [feature_names[i] for i in Indices[0:num]]

    
def reduce_features_bootstrap(DF, Branch_Names, num_reduced, num_iterations, num_estimators, learning_rate, max_depth):
    from sklearn.ensemble import GradientBoostingClassifier
    
    importances = np.zeros(len(Branch_Names))
    
    for i in range(num_iterations):
        train, test = make_test_train(DF, 'y', .666)
        BDT = GradientBoostingClassifier(n_estimators=num_estimators, learning_rate=learning_rate, max_depth=max_depth)
        BDT.fit(train[Branch_Names], train.y)
        importances += BDT.feature_importances_
    
    return [x for i,x in enumerate(Branch_Names) if i in np.argsort(BDT.feature_importances_)[::-1][0:num_reduced]]




#######################
# working with Graphs #
#######################


def add_information_from_tree(tree, matrix, weight=1):
    zipped = zip(tree.children_left, tree.children_right, tree.feature, tree.threshold)
    for i in range(len(zipped)):

        parent = zipped[i][2]
        left_child = zipped[zipped[i][0]][2]
        right_child = zipped[zipped[i][1]][2]
 
        if left_child >= 0:
            matrix[parent][left_child] += weight
        if right_child >= 0:
            matrix[parent][right_child] += weight
    
    return matrix

def get_directed_matrix(bdt, branch_names, weighted=False):
    connections = np.zeros((len(branch_names), len(branch_names)))
    if weighted:
        for i in range(len(bdt.estimators_)):
            connections = add_information_from_tree(bdt.estimators_[i].tree_, connections, weight = bdt.estimator_weights_[i])
    else:
        for tree in bdt.estimators_:
            connections = add_information_from_tree(tree.tree_, connections)
    return connections

def get_undirected_matrix(bdt, branch_names, weighted=False):
    connections = get_directed_matrix(bdt, branch_names, weighted)
    for i in range(len(connections)):
        for j in range(i):
            connections[i][j] += connections[j][i]
            connections[j][i] = 0
    return connections


def remove_edgeless(graph):
    degree_dict = graph.degree()
    to_keep = [x for x in degree_dict if degree_dict[x] > 0]
    return graph.subgraph(to_keep)




