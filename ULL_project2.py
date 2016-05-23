import argparse
import re
import numpy as np
import theano
import theano.tensor as T
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
    parser.add_argument('-L', metavar='learning_rate', dest="learning_rate",type=float,
                        help='initial learning rate')
    parser.add_argument('--half_after', metavar="iters", dest="ha",type=int,
                        help='learning halfs after this number of iters')
    parser.add_argument('--gru', dest="gru", action='store_true',
                        help='use this option to use gated units')
    parser.add_argument('-T', metavar='task', dest="task",type=str,
                        help='task to execute: [reverse,qa]')
    parser.add_argument('-V', metavar='valid_size', dest="valid_size",type=int,
                        help='percentage of training set used for validation')
    parser.add_argument('-C', metavar='cost_function', dest="cf",type=str,
                        help='cost funtion to use: \'ll\' - LogLikelihood (default) or \'ce\' - Cross Entropy')
    parser.add_argument('-l', metavar='lambda', dest="lmbd",type=float,
                        help='cost funtion regularization lambda')
    parser.add_argument('--random_weights', dest="wir", action='store_true',
                        help='use this option to use random initialization of weights (default is ones)')
    parser.add_argument('--qa_file', dest="qa_file", type=int,
                        help='for question answer task: specify task file to use (0 is all)')

    parser.set_defaults(embedding_size=12)
    parser.set_defaults(learning_rate=0.5)
    parser.set_defaults(iters=20)
    parser.set_defaults(ha=10)
    parser.set_defaults(gru=False)
    parser.set_defaults(task='reverse')
    parser.set_defaults(valid_size=0)
    parser.set_defaults(cf='ll')
    parser.set_defaults(lmbd=0.1)
    parser.set_defaults(wir=False)
    parser.set_defaults(qa_file=0)
    
    args = parser.parse_args()
    
    global nr_hidden
    nr_hidden = args.nr_hidden

    global embedding_size
    embedding_size = args.embedding_size

    global valid_size
    valid_size = args.valid_size

    global embed
    embed = False if  embedding_size==0 else True

    global iters
    iters = args.iters

    global reverse
    reverse = True if args.task=='reverse' else False

    global qa_file
    qa_file= args.qa_file

    test_set = 'test' if args.valid_size==0 else 'valid'
    
    print_info(args.gru,args.learning_rate,args.ha,args.cf,args.lmbd,test_set,valid_size,args.wir,qa_file)

    global C
    global data
    load_data()

    Y = run_model(args.gru, args.learning_rate,args.ha,args.cf,args.lmbd,test_set,args.wir)
    evaluate(Y, test_set)

def run_model(gru,lr,ha,cf,lmbd,test_set,wir):

    print('compiling theano computational graph ... ')
    voc_len = len(C.getVocabulary())
    
    m = Model(nr_hidden,embedding_size,voc_len,wir)
    if gru:
        encoder = GRUEncoder(m)
        decoder = GRUDecoder(m)
    else:
        encoder = Encoder(m)
        decoder = Decoder(m)

    if embed:
        embedding = Embedding(m)
        get_y = DeEmbed(m,embedding.get_parameters())
    
    x = T.imatrix()
    y = T.imatrix()

    if embed:
        x_e = embedding.get_output_expr(x)
        c=encoder.get_output_expr(x_e)
    else:
        c=encoder.get_output_expr(x)
    
    l = T.scalar(dtype='int32')

    if embed:
        y_e=decoder.get_output_expr(c,l)
        y_pred = T.nnet.softmax(get_y.get_output_expr(y_e))
    else:
        y_pred = T.nnet.softmax(decoder.get_output_expr(c,l))
        
    params = embedding.get_parameters() + encoder.get_parameters() + decoder.get_parameters() if embed else encoder.get_parameters() + decoder.get_parameters()
    
    cost = m.get_cost(y_pred,y,params,lmbd,cf)
    updates = m.get_sgd_updates(cost, params, lr)
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    print('training ... ')
    
    for i in range(iters):
        lr = lr/2 if i>=ha-1 else lr
        for x, y in zip(data['train']['input'], data['train']['output']):
            l = len(x) if reverse else 1
            y_pred, cost = trainF(x,y,l)

        print(' it: %d\t cost:\t%.5f'%(i+1,cost))

    Y = []

    print('\ntesting ... ')
    for x, y in zip(data[test_set]['input'], data[test_set]['output']):
        l = len(x) if reverse else 1
        y_pred, _ = test(x,y,l)
        #Y.append([np.argmax(y_pred[i]) for i in range(len(y_pred))])
        pred_sen = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        Y.append(C.getVocabulary()[pred_sen])

    
    return Y


def evaluate(pred_y, test_set):
    
    print('evaluating ...')
    original_input = C.getVectors(translated=False, reverse=reverse)
    check = 0
    tot = 0
    for a, b in zip(original_input[test_set]['output'], pred_y):
        for wa, wb in zip(a,b):
            tot +=1
            if not np.array_equal(wa,wb):
                check+=1
    percentage = (tot-check)*100.0/tot
    
    print('\nprecision: %.2f'%(percentage)+'%')
    
        
def load_data():
    print('loading data ...')
    
    fns = ['qa1_single-supporting-fact',
           'qa2_two-supporting-facts',
           'qa3_three-supporting-facts',
           'qa4_two-arg-relations',
           'qa5_three-arg-relations',
           'qa6_yes-no-questions',
           'qa7_counting',
           'qa8_lists-sets',
           'qa9_simple-negation',
           'qa10_indefinite-knowledge',
           'qa11_basic-coreference',
           'qa12_conjunction',
           'qa13_compound-coreference',
           'qa14_time-reasoning',
           'qa15_basic-deduction',
           'qa16_basic-induction',
           'qa17_positional-reasoning',
           'qa18_size-reasoning',
           'qa19_path-finding',
           'qa20_agents-motivations']
    
    if qa_file==0:
        if reverse:
            task_file_range = range(5)
        else:
            task_file_range = range(20)
    else:
        task_file_range = [qa_file-1]
        
    files = []
    for fn in np.array(fns)[task_file_range]:
        files.append('tasksv11/en/'+fn+'_train.txt')
        files.append('tasksv11/en/'+fn+'_test.txt')
        
    global C
    C = Collection(files, valid_size)
    C.printInfo()
    
    C.translate()

    global data
    data = C.getVectors(reverse=reverse, oneHot=True)

def print_info(gru,lr,ha,cf,lmbd,tset,vsize,wir,qa_file):
    cfs ='Cross Entropy' if cf=='ce' else  'Log Likelihod' 
    if vsize==0:
        testsets = 'test set'
    else:
        testsets= 'validation set ('+str(vsize)+'%)'

    wistr = 'random' if wir else 'ones'
    taskstr = 'reverse' if reverse else 'question answering - %d'%qa_file
    print( 'Network params on task %s:\n'\
           ' nr of iters:\t\t%d\n'\
           ' hidden nodes:\t\t%d\n'\
           ' embedding layer size:\t%d\n'\
           ' gated units used:\t%s\n'\
           ' initialization of weights:\t%s\n'\
           ' learning rate:\t\t%f\n'\
           ' lr half after:\t\t%d iters\n'\
           ' cost function:\t\t%s\n'\
           ' regularization lambda:\t%f\n'\
           ' results on %s\n'
          %(taskstr,iters,nr_hidden,embedding_size,gru,wistr,lr,ha,cfs,lmbd,testsets))
    
          
if __name__ == "__main__":
    main()
    
