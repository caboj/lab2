import argparse
import re
import numpy as np
import theano
import theano.tensor as T


def main():
    parser = argparse.ArgumentParser(description='invert sentence with RNN encoder decoder')
    parser.add_argument('-H', metavar='nr_hidden', dest="nr_hidden",type=int,
                   help='number of hidden nodes', required=True)

    global trainD
    global vecs

    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden
    load_data()
    train()

def load_data():
    fns = ['qa1_single-supporting-fact_train.txt',
           'qa2_two-supporting-facts_train.txt',
           'qa3_three-supporting-facts_train.txt',
           'qa4_two-arg-relations_train.txt',
           'qa5_three-arg-relations_train.txt']

    
    global trainD    
    trainD = []

    # split data into only lowercases words (remove punctiation and numbers) 
    for fn in fns:
        with open('tasksv11/en/'+fn) as f:
            trainD.extend(np.array([re.sub("[^a-zA-Z]", " " , l).lower().split() for l in  f.readlines()]))


    trainD = np.array(trainD)
    voc = np.unique(np.hstack(trainD)).tolist()
    voc.append('<BOS>')
    #print(voc)
    y = np.eye(len(voc))

    global vecs
    vecs = dict(zip(voc,y))

# copied from illctheanotutorial - modified for use here
def weights_init(shape):
    a = np.random.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return q

def embedding_init():
    return np.random.randn(1, len(vecs)) * 0.01

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
        self.W = theano.shared(weights_init((nr_hidden,len(vecs))))
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
        self.O = theano.shared(weights_init((len(vecs),nr_hidden)))
        self.V = theano.shared(weights_init((nr_hidden,nr_hidden)))
        self.Yh = theano.shared(weights_init((len(vecs),len(vecs))))


    
        W = T.matrix()
        W_in_1 = T.matrix()
        W_in_2 = T.matrix()
        W_feedback = T.matrix()
        W_out = T.matrix()

    def get_output_expr(self,h,l):
        #u = T.matrix() # it is a sequence of vectors
        h0 = h # initial state of x has to be a matrix, since
        c = h
        # it has to cover x[-3]
        y0 = theano.shared(vecs['<BOS>']) # y0 is just a vector since scan has only to provide
        # y[-1]
        

        ([h_vals, y_vals], updates) = theano.scan(fn=self.oneStep,
                                                  #sequences=[],
                                          outputs_info=[h0, y0],
                                          non_sequences=[c,self.O, self.V,self.Yh],
                                                  n_steps=l,
                                          strict=True)
        return y_vals
    
    def oneStep(self, h_tm1, y_tm1, c,O, V, Yh):

        h_t = T.tanh(theano.dot(V,h_tm1)+theano.dot(V,c))
        y_t = theano.dot(O,h_t)+theano.dot(Yh,y_tm1)+theano.dot(O,c)

        return [h_t, y_t]

    def get_parameters(self):
        return [self.W, self.U]


def train():
    #embedding_layer = EmbeddingLayer(embedding_init)
    encoder = Encoder()
    decoder = Decoder()
    x = T.imatrix()

    #embedding_expr = embedding_layer.get_output_expr(x)
    h=encoder.get_output_expr(x)
    encode = theano.function(inputs=[x],outputs=h)    

    l = T.scalar(dtype='int32')
    c = T.vector()
    y=decoder.get_output_expr(c,l)
    decode = theano.function(inputs=[c,l],outputs=y)
    
    for sen in trainD:
        x = np.array([vecs[sen[i]] for i in range(len(sen))],dtype=np.int32)
        H = encode(x)
        c = H[-1]
        y = decode(c,len(sen))
        #pred_rev_sen = out_to_words(y)

    print(y)

'''
def out_to_words(y):
    for outvec in y:
        softmax
'''
    
if __name__ == "__main__":
    main()
    
