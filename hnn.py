import glob
import os
import socket
import argparse

import numpy as np
import pylab as pl
import PIL as pil


def from_png_to_bipolar(name):
    """ This function takes image files and converts them into bipolar numbers.
    It returns the bipolar results of the corresponding picture.
    """ 
    # 10000 neurons, to store 100x100 pixels
    size = 100, 100 
    image = pil.Image.open(name).convert('L')
    resized_image = image.resize(size, pil.Image.ANTIALIAS)
    values = np.array(resized_image.getdata())
    sign = np.vectorize(lambda x: -1 if x < (255/2) else 1)
    return sign(values)


def train(patterns):
    """ This function trains with the Hebb training rule the Hopefield network.
    It returns the normalized weights.
    """
    pixcount, neurons  = patterns.shape # every pixel represents a neuron
    W = np.zeros((neurons, neurons)) # initialize weights
    for p in patterns: 
        W = W + np.outer(p, p) # adjust weights to reflect the correlation between pixels
    W[np.diag_indices(neurons)] = 0 # neurons are not connected to themselves, therefore set the main diagonal to weight 0 
    #print "%s" % (W)
    return W/pixcount 


def transfer():
    """ the transfer function or called sigmoid step function.
    returns a bipolar vector
    """
    return np.vectorize(lambda x: 1 if x >= 0 else -1)


def recover(W, patterns, iterations):
    """ This function tries to recover the blackened or noisy pictures from the trained ones.
    Information is propaged asynchronous, which is repeated for fixes number of iterations
    It returns the recovered patterns.
    """
    sign = transfer()

    for i in xrange(iterations): # asynchronous method, fixed number of iterations
        patterns = sign(np.dot(patterns, W)) # adjust the neuron activity to reflect the weights
    return patterns

def energy(W, patterns):
    return np.array([-0.5 * np.dot(np.dot(p.T, W), p) for p in patterns])


def main():
    """ This is the main function. It calls the above functions and generate
    the training PNGs and the suggested PNGs after a certain iteration step.
    One can use the underneath parameters to call the script. 
    """
    parser = argparse.ArgumentParser(description='Discrete Hopfield Network for Image Suggestions')
    parser.add_argument('-o', '--original', help='original PNG path', type=str, required=True)
    parser.add_argument('-c', '--cue', help='cue PNG path', type=str, required=True)
    parser.add_argument('-i', '--iterations', help='Type here the iteration count', type=int, required=True)
    args = vars(parser.parse_args())

    # from these files create bipolar matrices
    patterns = np.array([from_png_to_bipolar(args['original'])])
   
    # remember how large the patterns are
    side = int(np.sqrt(len(patterns[0]))) 
    
    # Train the network
    W = train(patterns)
    
       
    ''' Patterns with noise '''
    print "patterns with noise"
    interfere_patterns = np.array([from_png_to_bipolar(args['cue'])])
    
    # Four axes, returned as a 2-d array
    f, axarr = pl.subplots(2, len(patterns))
    axarr[0].set_title('cue')
    axarr[1].set_title('recover')
    axarr[0].imshow(interfere_patterns[0].reshape((side, side)), cmap="hot")
    axarr[0].axis('off')
    axarr[1].imshow(recover(W, interfere_patterns, args['iterations'])[0].reshape((side, side)), cmap="hot")
    axarr[1].axis('off')
    pl.suptitle('predict with interfere noise')
    pl.savefig('noisy-cue.png')
    

# main
if __name__ == '__main__':
    main()
