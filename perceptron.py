#!/usr/bin/python3

# AUTHOR:  Junkang Gu
# NetID:   jgu8
# csugID:  jgu8

import matplotlib.pyplot as plt #TODO COMMENT OUT THIS
import numpy as np


# Return tuple of feature vector (x, as an array) and label (y, as a scalar).
def parse_add_bias(line):
    tokens = line.split()
    x = np.array(tokens[:-1] + [1], dtype=np.float64)
    y = np.float64(tokens[-1])
    return x,y

# Return tuple of list of xvalues and list of yvalues
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_add_bias(line) for line in f]
        (xs,ys) = ([v[0] for v in vals],[v[1] for v in vals])
        return xs, ys


# Do learning.
def perceptron(train_xs, train_ys, iterations):
    weights = np.zeros(len(train_xs[0]))
    for i in range(iterations):
        samplecount = len(train_xs)

        for j in range(samplecount):
            y = activate(np.sum(train_xs[j]*weights))
            if y != train_ys[j]:
                weights = weights - y*train_xs[j]
      
    return weights

def perceptron_train1round(train_xs, train_ys, weights):
    samplecount = len(train_xs)
    for j in range(samplecount):
        y = activate(np.sum(train_xs[j]*weights))
        if y != train_ys[j]:
            weights = weights - y*train_xs[j]    
    return weights


# Return the accuracy over the data using current weights.
def test_accuracy(weights, test_xs, test_ys):
    counter = 0
    samplecount = len(test_xs)
    for i in range(samplecount):    
        if activate(np.sum(test_xs[i]*weights)) == test_ys[i]:
            counter = counter + 1
    return counter/samplecount

def activate(num):
    if num >= 0:
        return 1
    else:
        return -1

def find_mvn(train_xs):
    max = 0
    for x in train_xs:
        tmp = np.linalg.norm(x)
        if tmp > max:
            max = tmp
    return max

def find_delta(test_xs, test_ys, weights):
    delta = 0
    samplecount = len(test_xs)
    for i in range(samplecount):    
        x = np.sum(test_xs[i]*weights)*test_ys[i]
        if x<0:
            return -1
        if x>delta:
            delta = x
    return delta

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--train_file', type=str, default=None, help='Training data file.')
    parser.add_argument('--plot', type=bool, default=False, help='Print out all iterations until convergence')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.iterations: int; number of iterations through the training data.
    args.train_file: str; file name for training data.
    args.plot: bool; Whether test and print all iterations or not. Will terminate when reached 100% accuracy or ctrl-c interrupted by user
    """
    train_xs, train_ys = parse_data(args.train_file)
    print('Maximum vector norm:',find_mvn(train_xs))
    if args.plot:
        weights = np.zeros(len(train_xs[0]))
        iteration = 1
        data = [] #accuracy
        peak_accuracy = 0
        fig = plt.figure()
        while True:
            weights = perceptron_train1round(train_xs, train_ys, weights)
            accuracy = test_accuracy(weights, train_xs, train_ys)
            data.append(accuracy)
            if accuracy > peak_accuracy:
                peak_accuracy = accuracy
            if accuracy == 1:
                print('Delta=',find_delta(train_xs,train_ys,weights))
                print('Iteration:', iteration,'Accuracy:',accuracy,'Peak accuracy:',peak_accuracy)
                plt.plot(data)
                fig.savefig('plot.png')
                break
            if iteration % 10000 == 0:
                print('Iteration:', iteration,'Accuracy:',accuracy,'Peak accuracy:',peak_accuracy)
                plt.plot(data)
                fig.savefig('plot.png')
            iteration = iteration + 1


                
                
            
    else:
        weights = perceptron(train_xs, train_ys, args.iterations)
        accuracy = test_accuracy(weights, train_xs, train_ys)
        print('Train accuracy: {}'.format(accuracy))
        print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))


if __name__ == '__main__':
    main()
