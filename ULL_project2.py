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
                        help='word embedding layer size')
    parser.add_argument('-I', metavar='iters', dest="iters",type=int,
                        help='number of training iterations')
    parser.add_argument('--gru', dest="gru", action='store_true',
                        help='set to True if model should use gated units')
    parser.add_argument('-T', metavar='task', dest="task",type=str,
                        help='task to execute: [reverse,qa]')
    
    parser.set_defaults(embedding_size=0)
    parser.set_defaults(iters=5)
    parser.set_defaults(gru=False)
    parser.set_defaults(task='reverse')


    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden

    global embedding_size
    embedding_size = args.embedding_size

    global embed
    embed = False if  embedding_size==0 else True

    global iters
    iters = args.iters

    global reverse
    reverse = True if args.task=='reverse' else False
    
    global C
    load_data()
    
    Y = run_model(args.gru)

    evaluate(Y)

def run_model(gru):

    print('compiling theano computational graph ... ')
    voc_len = len(C.getVocabulary())
    
    m = Model(nr_hidden,embedding_size,voc_len)
    if gru:
        encoder = GRUEncoder(m)
        decoder = GRUDecoder(m)
    else:
        encoder = Encoder(m)
        decoder = Decoder(m)

    if embed:
        embedding = Embedding(m)
        get_y = DeEmbed(m)
    
    x = T.imatrix()
    y = T.imatrix()

    if embed:
        x_e = embedding.get_output_expr(x)
        c=encoder.get_output_expr(x_e)
    else:
        c=encoder.get_output_expr(x)
    
    l = T.scalar(dtype='int32')

    lr = 0.001

    if embed:
        y_e=decoder.get_output_expr(c,l)
        y_pred = T.nnet.softmax(get_y.get_output_expr(y_e))
    else:
        y_pred = T.nnet.softmax(decoder.get_output_expr(c,l))
        
    params = embedding.get_parameters() + encoder.get_parameters() + decoder.get_parameters() + get_y.get_parameters() if embed else encoder.get_parameters() + decoder.get_parameters()
    
    cost = m.get_cost(y_pred,y,params,.0000000001)
    updates = m.get_sgd_updates(cost, params, lr)
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    print('training ... ')

    for i in range(iters):
        lr = lr/2 if i>9 else lr
        for x, y in zip(trainD['input'], trainD['output']):
            l = len(x) if reverse else 1
            y_pred, cost = trainF(x,y,l)

                                              # INCOMPATIBLE WITH PYTHON 2.x
        print(' it: %d\t cost:\t%.5f'%(i,cost))#,end='\r')

    Y = []

    print('\ntesting ... ')
    for x, y in zip(testD['input'], testD['output']):
        l = len(x) if reverse else 1
        y_pred, _ = test(x,y,l)
        #Y.append([np.argmax(y_pred[i]) for i in range(len(y_pred))])
        pred_sen = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        Y.append(testC.getVocabulary()[pred_sen])
    
    return Y


def evaluate(pred_y):
    
    print('evaluating ...')
    original_input = testC.getVectors(translated=False, reverse=reverse)
    check = 0
    tot = 0
    for a, b in zip(original_input['output'], pred_y):
        for wa, wb in zip(a,b):
            tot +=1
            if not np.array_equal(wa,wb):
                check+=1
    percentage = (tot-check)*100.0/tot
    print('\nprecision: %.2f'%(percentage), '%')
    
        
def load_data():
    print('loading data ...')
    '''
    fns = ['qa1_single-supporting-fact',
           'qa2_two-supporting-facts',
           'qa3_three-supporting-facts',
           'qa4_two-arg-relations',
           'qa5_three-arg-relations']
    '''
    fns = ['qa1_single-supporting-fact']
    
    files = []
    for fn in fns:
        files.append('tasksv11/en/'+fn+'_train.txt')
        
    global C
    C = Collection(files)
    #C.printInfo()
    
    C.translate()

    global trainD
    trainD = C.getVectors(translated=False, reverse=reverse, oneHot=True)

    '''
    fns = ['qa1_single-supporting-fact',
           'qa2_two-supporting-facts',
           'qa3_three-supporting-facts',
           'qa4_two-arg-relations',
           'qa5_three-arg-relations']
    '''
    fns = ['qa1_single-supporting-fact']
    
    files = []
    for fn in fns:
        files.append('tasksv11/en/'+fn+'_test.txt')

    global testC
    testC = Collection(files)
    testC.translate()
    global testD
    testD = testC.getVectors(translated=False, reverse=reverse, oneHot=True)

    
if __name__ == "__main__":
    main()
    
