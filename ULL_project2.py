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


    args = parser.parse_args()

    global nr_hidden
    nr_hidden = args.nr_hidden
    global embedding_size
    embedding_size = args.embedding_size
    
    global C
    load_data()
    
    Y = run_model(args.gru)

def run_model(gru):
    voc_len = len(C.getVocabulary())
    
    m = Model(nr_hidden,embedding_size,voc_len)
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
    params = encoder.get_parameters() + decoder.get_parameters() + get_y.get_parameters()
    cost = m.get_cost(y_pred,y,params,.2)
    updates = m.get_sgd_updates(cost, params)
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    trainD = C.getVectors(translated=False, reverse=True, oneHot=True)
    for i in range(1):
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
        Y.append(testC.getVocabulary()[pred_sen])
    
    #print(Y)
    
    #'''
    # debug: print some information on and examples of output

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
    print('\n# errors: '+str(check))
    #'''
        
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
    
    C.translate()

if __name__ == "__main__":
    main()
    
