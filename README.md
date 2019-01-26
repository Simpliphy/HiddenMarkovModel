# HiddenMarkovModel
This is a first implementation of Hidden Markov Model. The implementation uses Expectation Maximization to find the 
transition matrix, emission parameters and the states distributions.
Only the gaussian emission is implemented.

# Anaconda environment

````
conda env create --file requirement.yml
````

# Examples

Script for the examples are in the folder examples.

# Tests

To run the tests

````
pytest -v tests/
```` 

# TODO

- vectorize the for-loops 
- Documentation for equations [Rabiner 1989](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
- More tests for good coverage
- Objects: model parameter G-HMM and training parameters (for cleaner code)

# Known issues
- Numerical instability in transition matrix for low visited states
- High sensibility to initial states calculated
- Does not penalize enough for high variance

