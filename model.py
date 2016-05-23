import theano
import theano.tensor as T
import numpy as np

class Model:

    def __init__(self, nr_hidden,embedding_size,voc_size,wir):
        self.nr_hidden = nr_hidden
        self.embedding_size = embedding_size
        self.voc_size = voc_size
        self.w_init_random = wir
        
    def weights_init(self,shape):
        if self.w_init_random:
            a = np.random.normal(0.0, 1.0, shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == shape else v
            q = q.reshape(shape)
            return q
        else:
            return np.ones(shape)

    def gates_init(self,size):
        return np.random.normal(0.0, 1.0, size)

    def get_sgd_updates(self,cost, params, lr=0.01):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - lr * g])
        return updates

    def get_cost(self,y_pred,y,params,lmbd,f='ll'):
        p = sum([T.square(par).sum() for par in params])
        if f=='ce':
            cost_function = lambda y_pred_w,y_w : T.nnet.categorical_crossentropy(y_pred_w,y_w)
        else:
            cost_function = lambda y_pred_w,y_w : -T.log(T.dot(y_pred,y_w))
        cost_w, _ = theano.scan(fn=cost_function,
                                sequences=[y_pred,y])

        return (1-lmbd)*T.sum(cost_w)+lmbd*p



class Embedding(Model):
    def __init__(self,m):
        self.E = theano.shared(m.weights_init((m.embedding_size,m.voc_size)))

    def get_output_expr(self, x):
        return T.dot(self.E,x.transpose()).transpose()

    def get_parameters(self):
        return [self.E]

class Encoder(Model):
    def __init__(self,m):
        
        in_size = m.voc_size if m.embedding_size==0 else m.embedding_size
        
        self.W = theano.shared(m.weights_init((m.nr_hidden,in_size)))
        self.U = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.V = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        
    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.U.shape[0], ))

        h, _ = theano.scan(fn=self.__get_rnn_step_expr,
                           sequences=input_sequence,
                           outputs_info=[h0])

        return T.tanh(T.dot(self.V,h[-1]))

    def __get_rnn_step_expr(self, x_t, h_tm1):
        return T.tanh(T.dot( self.U,h_tm1) + T.dot( self.W,x_t))

    def get_parameters(self):
        return [self.W, self.U, self.V]

class GRUEncoder(Model):
    def __init__(self,m):

        in_size = m.voc_size if m.embedding_size==0 else m.embedding_size
    
        self.W = theano.shared(m.weights_init((m.nr_hidden,in_size)))
        self.V = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.U = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Wr = theano.shared(m.weights_init((m.nr_hidden,in_size)))
        self.Ur = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Wz = theano.shared(m.weights_init((m.nr_hidden,in_size)))
        self.Uz = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.r = theano.shared(m.gates_init(m.nr_hidden))
        self.z = theano.shared(m.gates_init(m.nr_hidden))


    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.U.shape[0], ))

        h, _ = theano.scan(fn=self.__get_rnn_step_expr,
                           sequences=input_sequence,
                           outputs_info=[h0])

        return T.tanh(T.dot(self.V,h[-1]))

    def __get_rnn_step_expr(self, x_t, h_tm1):
        self.z = T.nnet.sigmoid(T.dot(self.Wz,x_t)+T.dot(self.Uz,h_tm1))
        self.r = T.nnet.sigmoid(T.dot(self.Wr,x_t)+T.dot(self.Ur,h_tm1))
        hj = T.tanh(T.dot(self.W,x_t)+T.dot(self.U,(self.r*h_tm1)))
        return self.z*h_tm1+(1-self.z)*hj

    def get_parameters(self):
        return [self.W, self.U,self.Wr, self.Ur,self.Wz, self.Uz]

class Decoder(Model):
    def __init__(self,m):
        out_size = m.voc_size if m.embedding_size==0 else m.embedding_size
        self.Ch = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Co = theano.shared(m.weights_init((out_size,m.nr_hidden)))
        self.Oh = theano.shared(m.weights_init((out_size,m.nr_hidden)))
        self.U = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.W = theano.shared(m.weights_init((m.nr_hidden,out_size)))
        self.Oy = theano.shared(m.weights_init((out_size,out_size)))

    def get_output_expr(self,c,l):
        h0 = T.tanh(T.dot(self.Ch,c))
        y0 = T.zeros((self.Oy.shape[0],))

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


class GRUDecoder(Model):
    def __init__(self,m):
        out_size = m.voc_size if m.embedding_size==0 else m.embedding_size
        self.W = theano.shared(m.weights_init((m.nr_hidden,out_size)))
        self.Wr = theano.shared(m.weights_init((m.nr_hidden,out_size)))
        self.Wz = theano.shared(m.weights_init((m.nr_hidden,out_size)))
        self.U = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Uz = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Ur = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.C = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Cz = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Cr = theano.shared(m.weights_init((m.nr_hidden,m.nr_hidden)))
        self.Oh = theano.shared(m.weights_init((out_size,m.nr_hidden)))
        self.Oy = theano.shared(m.weights_init((out_size,out_size)))
        self.Oc = theano.shared(m.weights_init((out_size,m.nr_hidden)))

    def get_output_expr(self,c,l):
        h0 = T.tanh(T.dot(self.C,c))
        y0 = T.zeros((self.Oy.shape[0],))

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

class DeEmbed(Model):
    def __init__(self,m,emb_mat):
        self.E = emb_mat[0]

    def get_output_expr(self,y_e):
        return T.dot(y_e,self.E)
        
    def get_parameters(self):
        return [self.E]
