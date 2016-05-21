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
    global vecs
    global word_to_idx
    global idx_to_word

    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden
    global embedding_size
    embedding_size = 40
    load_data()
    train()

def load_data_OLD():
    fns = ['qa1_single-supporting-fact_train.txt']
           #'qa2_two-supporting-facts_train.txt',
           #'qa3_three-supporting-facts_train.txt',
           #'qa4_two-arg-relations_train.txt',
           #'qa5_three-arg-relations_train.txt']

    
    global trainD    
    trainD = []

    # split data into only lowercases words (remove punctiation and numbers) 
    for fn in fns:
        with open('tasksv11/en/'+fn) as f:
            trainD.extend(np.array([re.sub("[^a-zA-Z]", " " , l).lower().split() for l in  f.readlines()]))


    trainD = np.array(trainD)
    voc = np.unique(np.hstack(trainD)).tolist()
    y = np.eye(len(voc))

    global vecs
    vecs = dict(zip(voc,y))
    global word_to_idx
    word_to_idx = dict(zip(voc,range(len(voc))))
    global idx_to_word
    idx_to_word = dict(zip(range(len(voc)),voc))
    
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

    C = Collection(files)
    C.printInfo()
    
    voc = C.getVocabulary()
    C.translate()

    vectors = C.getVectors(translated=False, reverse=True)
    global trainD
    trainD = vectors['input']

    y = np.eye(len(voc))
    global vecs
    vecs = dict(zip(voc,y))
    global word_to_idx
    word_to_idx = dict(zip(voc,range(len(voc))))
    global idx_to_word
    idx_to_word = dict(zip(range(len(voc)),voc))

# copied from illctheanotutorial - modified for use here
def weights_init(shape):
    a = np.random.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return q

def gates_init(size):
    return np.random.normal(0.0, 1.0, size)
    
def embedding_init():
    return np.random.randn(1, len(vecs)) * 0.01

# copied from illctheanotutorial - modified for use here
class EmbeddingLayer(object):
    def __init__(self, embedding_init):
        self.E = theano.shared(weights_init((nr_hidden,len(vecs))))

    def get_output_expr(self, input_expr):
        return self.embedding_matrix[input_expr]

    def get_parameters(self):
        return [self.embedding_matrix]

# copied from illctheanotutorial - RnnLayer
class Encoder(object):
    def __init__(self):
        self.W = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.U = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.V = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.E = theano.shared(weights_init((embedding_size,len(vecs))))
        
    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.U.shape[0], ))

        h, _ = theano.scan(fn=self.__get_rnn_step_expr,
                           sequences=input_sequence,
                           outputs_info=[h0])

        return T.tanh(T.dot(self.V,h[-1]))

    def __get_rnn_step_expr(self, x_t, h_tm1):
        return T.tanh(T.dot( self.U,h_tm1) + T.dot( self.W,T.dot(self.E,x_t)))

    def get_parameters(self):
        return [self.W, self.U, self.V,self.E]

class GRUEncoder(object):
    def __init__(self):
        self.E = theano.shared(weights_init((embedding_size,len(vecs))))
        self.W = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.V = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.U = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Wr = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.Ur = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Wz = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.Uz = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.r = theano.shared(gates_init(nr_hidden))
        self.z = theano.shared(gates_init(nr_hidden))


    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.U.shape[0], ))

        h, _ = theano.scan(fn=self.__get_rnn_step_expr,
                           sequences=input_sequence,
                           outputs_info=[h0])

        return T.tanh(T.dot(self.V,h[-1]))

    def __get_rnn_step_expr(self, x_t, h_tm1):
        x_e = T.dot(self.E,x_t)
        self.z = T.nnet.sigmoid(T.dot(self.Wz,x_e)+T.dot(self.Uz,h_tm1))
        self.r = T.nnet.sigmoid(T.dot(self.Wr,x_e)+T.dot(self.Ur,h_tm1))
        hj = T.tanh(T.dot(self.W,x_e)+T.dot(self.U,(self.r*h_tm1)))
        return self.z*h_tm1+(1-self.z)*hj

    def get_parameters(self):
        return [self.W, self.U,self.Wr, self.Ur,self.Wz, self.Uz, self.E]

class Decoder(object):
    def __init__(self):
        self.E = theano.shared((len(vecs),embedding_size))
        self.Ch = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Co = theano.shared(weights_init((embedding_size,nr_hidden)))
        self.Oh = theano.shared(weights_init((embedding_size,nr_hidden)))
        self.U = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.W = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.Oy = theano.shared(weights_init((embedding_size,embedding_size)))

    def get_output_expr(self,c,l):
        h0 = T.tanh(T.dot(self.Ch,c))
        y0 = theano.shared(np.zeros(embedding_size))

        ([h_vals, y_vals], updates) = theano.scan(fn=self.oneStep,
                                                  outputs_info=[h0, y0],
                                                  non_sequences=[c],
                                                  n_steps=l)
        return y_vals
    
    def oneStep(self, h_tm1, y_tm1, c):

        h_t = T.tanh(theano.dot(self.U,h_tm1)+theano.dot(self.Ch,c) + T.dot(self.W,y_tm1))
        y_e = theano.dot(self.Oh,h_t)+theano.dot(self.Oy,y_tm1)+theano.dot(self.Co,c)
        
        return [h_t, y_e]

    def get_parameters(self):
        return [self.Oh, self.Oy,self.U,self.Co,self.Ch]


class GRUDecoder(object):
    def __init__(self):

        self.W = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.Wr = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.Wz = theano.shared(weights_init((nr_hidden,embedding_size)))
        self.U = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Uz = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Ur = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.C = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Cz = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Cr = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Oh = theano.shared(weights_init((embedding_size,nr_hidden)))
        self.Oy = theano.shared(weights_init((embedding_size,embedding_size)))
        self.Oc = theano.shared(weights_init((embedding_size,nr_hidden)))
        
    def get_output_expr(self,c,l):
        h0 = T.tanh(T.dot(self.C,c))
        y0 = theano.shared(np.zeros(embedding_size))

        ([h_vals, y_vals], updates) = theano.scan(fn=self.oneStep,
                                                  outputs_info=[h0, y0],
                                                  non_sequences=[c],#,self.O, self.U,self.Yh,self.C,self.V],
                                                  n_steps=l)#,strict=True)
        return T.nnet.softmax(y_vals)
        
    def oneStep(self, h_tm1, y_tm1,c):#, c,O, U, Yh,C,V):
        z = T.nnet.sigmoid( T.dot(self.Wz,y_tm1) + T.dot(self.Uz,h_tm1) + T.dot(self.Cz,c) )
        r = T.nnet.sigmoid( T.dot(self.Wr,y_tm1) + T.dot(self.Ur,h_tm1) + T.dot(self.Cr,c) )
        hj = T.tanh( T.dot(self.W,y_tm1 + T.dot(r, (T.dot(self.U,h_tm1)+T.dot(self.C,c)) ) ))
        
        h_t = z*h_tm1+(1-z)*hj
        y_t = theano.dot(self.Oh,h_t)+theano.dot(self.Oy,y_tm1)+theano.dot(self.Oc,c)

        return [h_t, y_t]

    def get_parameters(self):
        return [self.W,self.Wr,self.Wz,self.U,self.Ur,self.Uz,self.C,self.Cr,self.Cz,self.Oy,self.Oh,self.Oc]

class ProbFromEmbed(object):
    def __init__(self):
        self.E = theano.shared(weights_init((embedding_size,len(vecs))))

    def get_output_expr(self,y_e):
        y = T.dot(y_e,self.E)
        return T.nnet.softmax(y)

    def get_parameters(self):
        return [self.E]
        
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
    get_y = ProbFromEmbed()
    
    x = T.imatrix()
    y = T.imatrix()

    c=encoder.get_output_expr(x)
    
    l = T.scalar(dtype='int32')
    y_e=decoder.get_output_expr(c,l)
    y_pred = get_y.get_output_expr(y_e)
    
    cost = get_cost(y_pred,y)
    updates = get_sgd_updates(cost, encoder.get_parameters() + decoder.get_parameters() + get_y.get_parameters())
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    for i in range(1):
        for sen in trainD:
            x = np.array([vecs[sen[i]] for i in range(len(sen))],dtype=np.int32)
            y = x[::-1]
            l = len(sen)
            y_pred, cost = trainF(x,y,l)
            #print('it: %d\t cost:%.5f'%(i,cost),end='\r')

    print()    
    Y = []
    for sen in trainD:
        x = np.array([vecs[sen[i]] for i in range(len(sen))],dtype=np.int32)
        y = x[::-1]
        l = len(sen)
        y_pred, _ = test(x,y,l)
        pred_sen = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        Y.append([idx_to_word[pred_w] for pred_w in pred_sen])

    print(Y)
    
    
if __name__ == "__main__":
    main()
    
