# Python Hopfield Network

This Python 2.7 script is an implementation of a discrete Hopfield network, which is capable to recover noisy
black and white ASCII pictures. It uses the following essential science Python libraries:

  *  NumPy - for large multidimensional arrays and matrices calculation
  *  PyLab - to plot the bipolar patterns into pictures

The nice thing is that it doesn't need any iteration for learning. 
It is just the outer product between input vector and the transposed input
vector, with 100x100 pixels (10000 neurons).

## Usage

  * -i to set the iteration runs
  * -o original picture
  * -c cue picture with the noise

One can play now with the different parameters to see how the Hopfield network
will recover noisy pictures.

## Example
As an example: here is the Mona Lisa with a mustache and the recoverd picture.

`$ python hnn.py -i 1 -o Mona_Lisa.png -c Mona_Lisa_Mustache.png`

![Original](https://github.com/berxter/python_hopfield_network/blob/master/Mona_Lisa.png)
![Mustache](https://github.com/berxter/python_hopfield_network/blob/master/Mona_Lisa_Mustache.png)
![Recovered](https://github.com/berxter/python_hopfield_network/blob/master/noisy-cue.png)
