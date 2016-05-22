import argparse
import re
import numpy as np
import theano
import theano.tensor as T
#import model as m
from model import *
from corpus import *

def main():
    parser = argparse.ArgumentParser(description='invert sentence with RNN encoder decoder')
    parser.add_argument('-H', metavar='nr_hidden', dest="nr_hidden",type=int,
                        help='number of hidden nodes', required=True)
    parser.add_argument('-E', metavar='embedding_size', dest="embedding_size",type=int,
                        help='word embedding layer size', required=True)
    parser.add_argument('--gru', dest="gru", action='store_true',
                        help='set to True if model should use gated units')
    parser.set_defaults(gru=False)

    global trainD
    global vecs
    global word_to_idx
    global idx_to_word

    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden
    global embedding_size
    embedding_size = args.embedding_size
    load_data()
    Y = run_model(args.gru)

def run_model(gru):

    m = Model(nr_hidden,embedding_size,len(vecs))
    if gru:
        encoder = GRUEncoder(m)
        decoder = GRUDecoder(m)
    else:
        encoder = Encoder(m)
        decoder = Decoder(m)

    get_y = ProbFromEmbed(m)
    
    x = T.imatrix()
    y = T.imatrix()

    c=encoder.get_output_expr(x)
    
    l = T.scalar(dtype='int32')
    y_e=decoder.get_output_expr(c,l)
    y_pred = get_y.get_output_expr(y_e)
    
    cost = m.get_cost(y_pred,y)
    updates = m.get_sgd_updates(cost, encoder.get_parameters() + decoder.get_parameters() + get_y.get_parameters())
    
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

if __name__ == "__main__":
    main()
    
