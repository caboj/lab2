import argparse
import re
import numpy as np
import theano
import theano.tensor as T
from corpus import *

def main():
    parser = argparse.ArgumentParser(description='invert sentence with RNN encoder decoder')
    parser.add_argument('-H', metavar='nr_hidden', dest="nr_hidden",type=int,
                   help='number of hidden nodes', required=True)

    global trainD
    global C
    global voc_len

    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden
    load_data()
    train()

def load_data():
    '''
    fns = ['qa1_single-supporting-fact',
           'qa2_two-supporting-facts',
           'qa3_three-supporting-facts',
           'qa4_two-arg-relations',
           'qa5_three-arg-relations']
    #'''
    fns = ['qa1_single-supporting-fact']
    
    files = []
    for fn in fns:
        files.append('tasksv11/en/'+fn+'_train.txt')
        #files.append('tasksv11/en/'+fn+'_test.txt')

    global C
    C = Collection(files)
    C.printInfo()
    
    global voc_len
    voc = C.getVocabulary()
    voc_len = len(voc)

    C.translate()

    vectors = C.getVectors(translated=False, reverse=True, oneHot=True)
    global trainD
    trainD = vectors #['input']

# copied from illctheanotutorial - modified for use here
def weights_init(shape):
    a = np.random.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return q

def embedding_init():
    return np.random.randn(1, voc_len) * 0.01

# copied from illctheanotutorial - modified for use here
class EmbeddingLayer(object):
    def __init__(self, embedding_init):
        self.embedding_matrix = theano.shared(embedding_init())

    def get_output_expr(self, input_expr):
        return self.embedding_matrix[input_expr]

    def get_parameters(self):
        return [self.embedding_matrix]

# copied from illctheanotutorial - RnnLayer
class Encoder(object):
    def __init__(self):
        self.W = theano.shared(weights_init((nr_hidden,voc_len)))
        self.U = theano.shared(weights_init((nr_hidden,nr_hidden)))

    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.U.shape[0], ))

        h, _ = theano.scan(fn=self.__get_rnn_step_expr,
                           sequences=input_sequence,
                           outputs_info=[h0])
        return h

    def __get_rnn_step_expr(self, x_t, h_tm1):
        return T.tanh(T.dot( self.U,h_tm1) + T.dot( self.W,x_t))

    def get_parameters(self):
        return [self.W, self.U]

class Decoder(object):
    def __init__(self):
        self.O = theano.shared(weights_init((voc_len,nr_hidden)))
        self.V = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Yh = theano.shared(weights_init((voc_len,voc_len)))

    def get_output_expr(self,h,l):
        #u = T.matrix() # it is a sequence of vectors
        h0 = h # initial state of x has to be a matrix, since
        c = h
        # it has to cover x[-3]
        y0 = theano.shared(np.zeros(voc_len)) # y0 is just a vector since scan has only to provide
        # y[-1]
        

        ([h_vals, y_vals], updates) = theano.scan(fn=self.oneStep,
                                                  #sequences=[],
                                          outputs_info=[h0, y0],
                                          non_sequences=[c,self.O, self.V,self.Yh],
                                                  n_steps=l,
                                          strict=True)
        return T.nnet.softmax(y_vals)
    
    def oneStep(self, h_tm1, y_tm1, c,O, V, Yh):

        h_t = T.tanh(theano.dot(V,h_tm1)+theano.dot(V,c))
        y_t = theano.dot(O,h_t)+theano.dot(Yh,y_tm1)+theano.dot(O,c)

        return [h_t, y_t]

    def get_parameters(self):
        return [self.O, self.V,self.Yh]

def get_sgd_updates(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - lr * g])
    return updates

def get_cost(y_pred,y):

    cost_w, _ = theano.scan(fn=lambda y_pred_w,y_w : T.nnet.categorical_crossentropy(y_pred_w,y_w),
                            sequences=[y_pred,y])
    

    return T.sum(cost_w)

def train():
    
    encoder = Encoder()
    decoder = Decoder()

    x = T.imatrix()
    y = T.imatrix()

    h=encoder.get_output_expr(x)
    
    l = T.scalar(dtype='int32')
    y_pred=decoder.get_output_expr(h[-1],l)
        
    cost = get_cost(y_pred,y)
    updates = get_sgd_updates(cost, encoder.get_parameters() + decoder.get_parameters())
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    for i in range(10):
        for x, y in zip(trainD['input'], trainD['output']):
            l = len(x)
            y_pred, cost = trainF(x,y,l)

            # INCOMPATIBLE WITH PYTHON 2.x
            #print('it: %d\t cost:%.5f'%(i,cost),end='\r')

    print()    
    Y = []

    # debug: seperate test set
    '''
    testC = Collection(['tasksv11/en/qa1_single-supporting-fact_test.txt'])
    testC.translate()
    testD = testC.getVectors(translated=False, reverse=True, oneHot=True)
    #'''

    # debug: training set = test set
    #'''
    testC = C
    testD = trainD 
    #'''

    for x, y in zip(testD['input'], testD['output']):
        l = len(x)
        y_pred, _ = test(x,y,l)
        pred_sen = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        Y.append(C.getVocabulary()[pred_sen])
    
    #print(Y)
    
    original_input = testC.getVectors(translated=False, reverse=True)

    for i in range(5):
        #print trainD[i]
        print('\n')
        print('----------------------------------')
        print('IN')
        print(original_input['input'][i])
        print('\nOUT')
        print(original_input['output'][i])
        print('\nRESULT')
        print(Y[i])
        print('----------------------------------')
        #print ''
        #print 'check: '+str(len(original_input['output'][i])==len(Y[i]))

    check = 0
    for a, b in zip(original_input['output'], Y):
        if not np.array_equal(a,b):
            check+=1
    print('')
    print(check)
    
if __name__ == "__main__":
    main()
    
