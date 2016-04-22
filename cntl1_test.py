#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy
import copy
import ntm
from ntm import *
from addressing import writtenMemory,refocus,memOp ,betaSimilarity
from cntl1 import controller1
from math_utils import *
from addressing_test import doAddressing

def TestLogisticModel():
    times = 10
    #numpy.random.rand(times,4)
    x=numpy.array([[0.60,0.94,0.66,0.44],[0.42,0.69,0.07,0.16],[0.10,0.30,0.52,0.81],[0.21,0.38,0.32,0.47],[0.28,0.29,0.68,0.22],[0.20,0.36,0.57,0.86],[0.29,0.30,0.75,0.21],[0.87,0.70,0.52,0.03],[0.16,0.61,0.98,0.08],[0.59,0.06,0.69,0.30]])
    y=numpy.array([[0.17,0.54,0.54,0.28],[0.42,0.53,0.25,0.28],[0.79,0.36,0.88,0.30],[0.89,0.10,0.98,0.07],[0.22,0.68,0.24,0.31],[0.93,0.74,0.80,0.73],[0.18,0.43,0.90,0.68],[0.98,0.92,0.09,0.49],[0.93,0.95,0.35,0.69],[0.71,0.56,0.65,0.55]])

    n = 3
    m = 2
    h1Size = 3
    numHeads = 2
    c = controller1()
    c.NewEmptyController1(len(x[0]), len(y[0]), h1Size, numHeads, n, m)
    global c_weights_list
    c_weights_list=[1.51,0.808,0.261,1.97,1.79,0.644,1.44,1.29,0.171,1.34,1.25,0.739,0.433,1.11,0.804,1.01,0.337,0.663,1.66,1.4,0.116,2,0.823,0.223,1.56,0.184,0.107,1.43,0.502,1.7,1.95,0.425,0.0431,1.89,0.186,1.29,0.624,0.897,0.974,0.474,1.07,0.374,0.478,1.26,0.254,0.563,0.821,0.87,1.25,1.1,1.25,1.46,1.66,0.00103,1.47,0.8,0.996,1.21,0.819,0.0593,0.00381,0.00569,1.83,1.18,1.12,1.63,1.76,0.917,1.2,0.0525,1.69,0.499,1.28,0.495,0.347,1.19,1.63,1.39,0.0606,1.08,1.95,1.5,0.588,1.51,0.302,0.712,1.66,0.464,1.26,0.997,0.18,0.0504,0.784,1.18,1.86,1.14,1.18,0.824,1.11,0.983,1.92,1.59,0.215,1.57,0.787,0.261,0.38,1.48,1.31,0.197,1.04,0.199,0.304,0.152,0.63,0.319,0.276,0.645,1.08,1.14,1.03,1.37,1.31,1.05,1.31,1.43,1.27,0.0257,0.0614,0.196,0.738,1.65,0.695,0.689,0.506]
    c.Weights(c_weights_list)
    #test_weight,test_weightgrad = c.get_Weights()
    #print test_weight
    model = LogisticModel(Y=y)
    ForwardBackward(c,x,model)
    checkGradients(c, Controller1Forward, x, model)

def TestMultinomialModel():
    times = 10
    #numpy.random.rand(times,4)
    x=numpy.array([[0.604660,0.940509,0.664560,0.437714],[0.424637,0.686823,0.065637,0.156519],[0.096970,0.300912,0.515213,0.813640],[0.214264,0.380657,0.318058,0.468890],[0.283034,0.293102,0.679085,0.218553],[0.203187,0.360871,0.570673,0.862491],[0.293114,0.297083,0.752573,0.206583],[0.865335,0.696719,0.523820,0.028303],[0.158328,0.607253,0.975242,0.079454],[0.594809,0.059121,0.692025,0.301523]])
    
    outputSize = 4
    y=numpy.array([1,2,2,3,1,3,2,0,3,1])

    n = 3
    m = 2
    h1Size = 3
    numHeads = 2
    
    c = controller1()
    c.NewEmptyController1(len(x[0]), outputSize, h1Size, numHeads, n, m)
    global c_weights_list
    c_weights_list=[1.761086,0.594225,1.788723,0.194909,1.953834,0.148582,0.444579,1.362157,0.483030,0.623045,1.865693,1.483698,1.602110,1.460463,0.365850,0.856714,1.793984,1.365307,1.957859,1.844425,0.181675,0.986284,1.853974,1.909891,0.695908,1.381678,1.421814,1.127559,1.298979,1.103530,1.511647,0.807607,0.261302,1.971929,1.792683,0.644168,1.442296,1.289080,0.171041,1.339151,1.245457,0.739386,0.473645,1.070564,0.374492,0.477681,1.256196,0.253506,0.562661,0.820646,0.869825,1.250190,1.100294,1.247218,1.458361,1.661068,0.001028,1.472137,0.799968,0.995736,1.207956,0.819237,0.059343,0.003808,0.005686,1.831643,1.179668,1.118785,1.630810,1.756024,0.916885,1.200331,0.052530,1.691666,0.499386,1.283569,0.494933,0.347312,1.185248,1.628789,1.387676,0.060645,1.078420,1.951350,1.501526,0.588013,1.506323,0.301928,0.711535,1.663862,0.463660,1.255669,0.996789,0.179672,0.050388,0.784432,1.178766,1.859223,1.144174,1.177153,0.823525,1.105161,0.983215,1.915908,1.594417,0.214762,1.566070,0.786502,0.260828,0.380066,1.479652,1.308083,0.196768,1.040761,0.199459,0.303687,0.152381,0.630416,0.319302,0.275608,0.645221,1.078149,1.141703,1.025564,1.368350,1.306080,1.049000,1.308540,1.432737,1.273288,0.025652,0.061364,0.196062,0.738223,1.652908]
    c.Weights(c_weights_list)
    #test_weight,test_weightgrad = c.get_Weights()
    #print test_weight
    model = MultinomialModel(Y=y)
    ForwardBackward(c,x,model)
    #weights_list_temp,weights_grad_list_temp = c.get_Weights()
    #print weights_list_temp
    #print weights_grad_list_temp
    checkGradients(c, Controller1Forward, x, model)

def TestLinearModel():
    times = 10
    #numpy.random.rand(times,4)
    x=numpy.array([[10, 10],[10, 11],[10, 12],[10, 13],[10, 14],[10, 15],[10, 16],[10, 17]])
    y=numpy.array([[1],[2],[3],[1],[2],[3],[1],[2]])

    n = 3
    m = 2
    h1Size = 3
    numHeads = 2
    c = controller1()
    c.NewEmptyController1(len(x[0]), len(y[0]), h1Size, numHeads, n, m)
    print c.NumWeights()
    global c_weights_list
    c_weights_list=[1.51,0.808,0.261,1.97,1.79,0.644,1.44,1.29,0.171,1.34,1.25,0.739,0.433,1.11,0.804,1.01,0.337,0.663,1.66,1.4,0.116,2,0.823,0.223,1.56,0.184,0.107,1.43,0.502,1.7,1.95,0.425,0.0431,1.89,0.186,1.29,0.624,0.897,0.974,0.474,1.07,0.374,0.478,1.26,0.254,0.563,0.821,0.87,1.25,1.1,1.25,1.46,1.66,0.00103,1.47,0.8,0.996,1.21,0.819,0.0593,0.00381,0.00569,1.83,1.18,1.12,1.63,1.76,0.917,1.2,0.0525,1.69,0.499,1.28,0.495,0.347,1.19,1.63,1.39,0.0606,1.08,1.95,1.5,0.588,1.51,0.302,0.712,1.66,0.464,1.26,0.997,0.18,0.0504,0.784,1.18,1.86,1.14,1.18,0.824,1.11,0.983,1.92,1.59,0.215,1.57,0.787,0.261,0.38,1.48,1.31,0.197,1.04,0.199,0.304,0.152,0.63,0.319,0.276]
    c.Weights(c_weights_list)
    ##test_weight,test_weightgrad = c.get_Weights()
    ##print test_weight
    model = LinearModel(Y=y)
    ForwardBackward(c,x,model)
    checkGradients(c, Controller1Forward, x, model)

def Controller1Forward(c1 , reads , x ) :
    c = c1
    #print c.Wh1rVal
    #print reads
    Wh1rdotread = numpy.zeros(c.Wh1bVal.shape)
    for i in xrange(c.Wh1rVal.shape[0]):
        v=0
        for (j, read) in enumerate(reads):
            #print read.TopVal
            v += numpy.dot(c.Wh1rVal[i][j],reads[j,:].T)
            #print v
        Wh1rdotread[0][i] = v
    h1 = Sigmoid(numpy.dot(c.Wh1xVal,x)+Wh1rdotread+c.Wh1bVal)
    #print h1
    prediction = numpy.dot(h1,c.Wyh1Val[:,0:h1.shape[1]].T) + c.Wyh1Val[:,h1.shape[1]:].T
    #print prediction
    numHeads = c.Wh1rVal.shape[1]
    m = c.Wh1rVal.shape[2]
    heads = []
    for i in xrange(c.Wuh1Val.shape[0]):
        head_temp = Head()
        head_temp.NewHead(m)
        Wuh1ValdotH1 =  numpy.dot(h1,c.Wuh1Val[i,:,0:h1.shape[1]].T) + c.Wuh1Val[i,:,h1.shape[1]:].T
        #print Wuh1ValdotH1
        #print Wuh1ValdotH1[0,0]
        for j in xrange(head_temp.headUnitsLen()):
            head_temp.unitsVals[j] = Wuh1ValdotH1[0,j]
        heads.append(head_temp)
        #print head_temp.unitsVals
    return prediction, heads

def loss(c , forward , in_val , model ):
    mem = c.Mtm1BiasV()
    #print mem.N
    #print mem.TopVal.shape
    wtm1Bs = c.Wtm1BiasV()
    wtm1s = []
    for i in xrange(c.NumHeads()):
        TopVal = numpy.zeros((1,c.MemoryN()))
        W_temp = refocus(TopVal)
        sum  = 0
        for j in xrange(len(wtm1Bs[i])):
            W_temp.TopVal[0][j] = numpy.exp(wtm1Bs[i][j].TopVal[0][0])
            sum += W_temp.TopVal[0][j]
        W_temp.TopVal = W_temp.TopVal/sum
        #print W_temp.TopVal
        wtm1s.append(W_temp)
    reads = numpy.zeros((c.NumHeads(), c.MemoryM()))
    for i in xrange(reads.shape[0]):
        for j in xrange(reads.shape[0]):
            v=numpy.dot(wtm1s[i].TopVal,mem.TopVal[:,j])
            reads[i][j] = v
    #print reads
    prediction = []
    for t in xrange(len(in_val)):
        #forward(c, reads, in_val[t])
        prediction_t, heads = forward(c, reads, in_val[t])
        prediction_t = computeDensity(t, prediction_t, model)
        #print prediction_t
        prediction.append(prediction_t)
        for (i,head) in enumerate(heads):
            head.Wtm1 = wtm1s[i]
        wsFloat64, readsFloat64, memFloat64 = doAddressing(heads, mem)
        #print wsFloat64
        #print readsFloat64
        #print memFloat64
        wtm1s = transformWSFloat64(wsFloat64)
        reads = readsFloat64
        mem = transformMemFloat64(memFloat64)
        #print mem
    #print prediction
    return model.Loss(prediction)

def computeDensity(timestep , pred , model ):
    #print pred
    unitsVal = copy.deepcopy(pred)
    yHs_list=[unitsVal,None]
    model.Model(timestep, yHs_list=yHs_list)
    #print yHs_list[0]
    return yHs_list[0]

def checkGradients(c , forward , in_val , model ):
    lx = loss(c, forward, in_val, model)
    #print lx
    for (i,weight_val) in enumerate(c_weights_list):
        x = weight_val
        h = machineEpsilonSqrt * max(numpy.abs(x), 1)
        xph = x + h
        c_weights_list[i] = xph
        c.Weights(c_weights_list)
        lxph = loss(c, forward, in_val, model)
        c_weights_list[i] = x
        grad = (lxph - lx) / (xph - x)
        weights_list_temp,weights_grad_list_temp = c.get_Weights()
        if numpy.abs(grad-weights_grad_list_temp[i]) > 1e-5 :
            print "wrong ",i," gradient expected ",grad, "got ",  weights_grad_list_temp[i]
        else:
            print "OK ",i," gradient ",grad," ", weights_grad_list_temp[i]

def transformMemFloat64(memFloat64 ):
    memory = writtenMemory(N=memFloat64.shape[0],TopVal=memFloat64)
    return memory

def transformWSFloat64(wsFloat64 ):
    wtm1s = []
    for i in xrange(wsFloat64.shape[0]):
        TopVal = numpy.zeros((1,wsFloat64.shape[1]))
        W_temp = refocus(TopVal)
        sum  = 0
        for j in xrange(wsFloat64.shape[1]):
            W_temp.TopVal[0][j] = wsFloat64[i][j]
        #print W_temp.TopVal
        wtm1s.append(W_temp)
    return wtm1s

if __name__ == "__main__":
    #TestLogisticModel()
    #TestMultinomialModel()
    TestLinearModel()