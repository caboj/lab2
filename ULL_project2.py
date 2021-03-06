import argparse
import re
import numpy as np
import theano
import theano.tensor as T
from model import *
from corpus import *
import os

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
    parser.add_argument('--ones_init', dest="wir", action='store_false',
                        help='use this option to use random initialization of weights (default is ones)')
    parser.add_argument('--qa_file', dest="qa_file", type=int,
                        help='for question answer task: specify task file to use (0 is all)')
    parser.add_argument('--save_file', dest="save_file", type=str,
                        help='file for saving output')

    parser.set_defaults(embedding_size=12)
    parser.set_defaults(learning_rate=0.5)
    parser.set_defaults(iters=20)
    parser.set_defaults(ha=100)
    parser.set_defaults(gru=False)
    parser.set_defaults(task='reverse')
    parser.set_defaults(valid_size=0)
    parser.set_defaults(cf='ll')
    parser.set_defaults(lmbd=0.1)
    parser.set_defaults(wir=True)
    parser.set_defaults(qa_file=0)
    parser.set_defaults(save_file='')

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
    
    global save_file
    save_file = args.save_file
    if not os.path.exists('output/'+save_file):
        os.makedirs('output/'+save_file)


    info = print_info(args.gru,args.learning_rate,args.ha,args.cf,args.lmbd,test_set,valid_size,args.wir,qa_file)
    if (save_file):
        with open('output/'+save_file+'/'+save_file+'.txt', 'w+') as f:
            f.write(info)
            f.write('\n')

    global C
    global data
    load_data()

    run_model(args.gru, args.learning_rate,args.ha,args.cf,args.lmbd,test_set,args.wir)

    '''
    Y_train, Y_test = run_model(args.gru, args.learning_rate,args.ha,args.cf,args.lmbd,test_set,args.wir)
    print('evaluating ...')
    evaluate(Y_train, 'train')
    evaluate(Y_test, test_set)
    '''

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
    '''
    if cf=='ll':
        z = 1/y.shape[0]
        cost = (1-lmbd)*T.sum(-T.log(T.dot(T.dot(y_pred,y.transpose())),T.eye(z)))+(lmbd/z)*sum([T.square(par).sum() for par in params])
    elif cf=='ce':
        cost = m.get_cost(y_pred,y,params,lmbd,cf)
    #'''
        
    updates = m.get_sgd_updates(cost, params, lr)
    
    trainF = theano.function(inputs=[x,y,l],outputs=[y_pred,cost],updates=updates)
    
    test = theano.function(inputs=[x,y,l],outputs=[y_pred,cost])

    print('training ... ')
    
    for i in range(iters):
        lr = lr/2 if i>=ha-1 else lr
        for x, y in zip(data['train']['input'], data['train']['output']):
            l = len(x) if reverse else 1
            y_pred, cost = trainF(x,y,l)
            #print(y_pred,end='\r')
            #print('cost:\t%.5f'%(cost),end='\r')

        #print(' it: %d\t cost:\t%.5f'%(i+1,cost))
        #print('testing ... ')

        s1 = evaluate(testOutput('train', test, i), 'train')
        s2 = evaluate(testOutput(test_set, test, i), test_set)
        print('%d\t%s\t%s'%(i+1, str(s1['total']), str(s2['total'])))

        if save_file:
            with open('output/'+save_file+'/'+save_file+'.txt', 'a') as f:
                f.write(str(i+1)+'\n')
                f.write(str(s1)+'\n')
                f.write(str(s2)+'\n')

    '''
    print('\ntesting ... ')
    Y_train = testOutput('train', test)
    Y_test = testOutput(test_set, test)
    return Y_train, Y_test
    '''

def testOutput(data_set, test, it):
    Y = []
    for x, y in zip(data[data_set]['input'], data[data_set]['output']):
        l = len(x) if reverse else 1
        y_pred, _ = test(x,y,l)
        pred_sen = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
        Y.append(C.getVocabulary()[pred_sen])
    
    if save_file:
        with open('output/'+save_file+'/'+save_file+'_'+data_set+'_output_'+str(it+1)+'.txt', 'w+') as f:
            for l in Y:
                f.write(' '.join(l))
                f.write('\n')
    
    return Y

def evaluate(pred_y, data_set):
    values = {}
    values['total'] = evaluateSet(pred_y,data_set,list(range(len(pred_y))))
    lens = np.array([len(y) for y in pred_y])
    for l in np.sort(np.unique(lens)):
        idx = np.where(lens==l)[0]
        values[l] = evaluateSet(pred_y,data_set,idx)
    return values

def evaluateSet(pred_y, data_set, idx):
    original_input = C.getVectors(translated=False, reverse=reverse)
    check = 0
    tot = 0
    original = np.array(original_input[data_set]['output'])
    pred = np.array(pred_y)
    for a, b in zip(original[idx], pred[idx]):
        for wa, wb in zip(a,b):
            tot +=1
            if not np.array_equal(wa,wb):
                check+=1
    precision = round((tot-check)*100.0/tot, 5)
    #print('\tevaluation of '+data_set+' of size '+str(len(pred_y))+':')
    #print('\t\tprecision: %.2f'%(percentage)+'%')
    return precision
        
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

    if save_file:
        with open('output/'+save_file+'/'+save_file+'.txt', 'a') as f:
            f.write(str(list(np.array(fns)[task_file_range])))
            f.write('\n\n')
        
    global C
    C = Collection(files, valid_size)
    C.printInfo()
    
    C.translate()

    global data
    data = C.getVectors(reverse=reverse, oneHot=True)

    original_input = C.getVectors(translated=False, reverse=reverse)
    if save_file:
        writeDataset(original_input, 'train', 'input')
        if valid_size:
            writeDataset(original_input, 'valid', 'input')
        else:
            writeDataset(original_input, 'test', 'input')

def writeDataset(data, data_set, input_type):
    with open('output/'+save_file+'/'+save_file+'_'+data_set+'_'+input_type+'.txt', 'w+') as f:
        for l in data[data_set][input_type]:
            f.write(' '.join(l))
            f.write('\n')

def print_info(gru,lr,ha,cf,lmbd,tset,vsize,wir,qa_file):
    cfs ='Cross Entropy' if cf=='ce' else  'Log Likelihod' 
    if vsize==0:
        testsets = 'test set'
    else:
        testsets= 'validation set ('+str(vsize)+'%)'

    wistr = 'random' if wir else 'ones'
    taskstr = 'reverse' if reverse else 'question answering - %d'%qa_file
    info= ('Network params on task %s:\n'\
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
    print(info)
    return info
          
if __name__ == "__main__":
    main()
    
