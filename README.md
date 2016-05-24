ULL lab2

The program is run with 'python ULL_project2.py'. It is compatible with both python 2.x and 3.y. The following parameters are accepted:

-H		The number of hidden layers to use

-E		The embedding size to use

-I		The number of iterarations to run

-L		The learning rate

-T		Type of task: reverse (default) or qa

-V		Percentage of training set to use for validation. 0 for test set.

-C		Cost function to use: ll or ce

-l		Regularization parameter lambda

--ones_init	Initialize weight with ones instead of randomly 

--qa_file	Use single file instead of first 5 for reverse and all for qa

--save_file	Name of experiment to use in generated files. 

--half_after	The number of iterations after which to half the learning rate

--gru		Use GRU instead of RNN

EXAMPLES:

python ULL_project2.py -H 32 -E 40 -l 0.005 -L .001 --gru -I 120 --save_file 'test'

python ULL_project2.py -H 64 -E 40 -l 0.005 -L .001 --half_after 50 -I 100 -T qa -V 30 -C ce 
