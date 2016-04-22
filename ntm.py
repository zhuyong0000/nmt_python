#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy
import copy
from addressing import contentAddressing,refocus,memRead,memOp
from math_utils import *

class Head(object):
    def __init__(self):
        self.Wtm1=None
    def NewHead(self,m):
        self.unitsVals=numpy.zeros(3*m+4)
        self.unitsGrads=numpy.zeros(3*m+4)
        self.M=m
    def EraseVal(self):
        return self.unitsVals[0:self.M].reshape(1,self.M)
    def EraseGrad(self,k=None,val=None):
        if k is not None:
            assert k >= 0 and k < self.M
            if val is not None:
                self.unitsGrads[k]=val
            return self.unitsGrads[k]
        else:
            return self.unitsGrads[0:self.M].reshape(1,self.M)
    def AddVal(self):
        return self.unitsVals[self.M : 2*self.M].reshape(1,self.M)
    def AddGrad(self,k=None,val=None):
        if k is not None:
            assert k >= 0 and k < self.M
            if val is not None:
                self.unitsGrads[self.M + k]=val
            return self.unitsGrads[self.M + k]
        else:
            return self.unitsGrads[self.M : 2*self.M].reshape(1,self.M)
    def KVal(self):
        return self.unitsVals[2*self.M : 3*self.M].reshape(1,self.M)
    def KGrad(self,k=None,val=None):
        if k is not None:
            assert k >= 0 and k < self.M
            if val is not None:
                self.unitsGrads[2*self.M + k]=val
            return self.unitsGrads[2*self.M + k]
        else:
            if val is not None:
                self.unitsGrads[2*self.M : 3*self.M]=val[0]
            return self.unitsGrads[2*self.M : 3*self.M].reshape(1,self.M)
    def BetaVal(self,val=None):
        if val is not None:
            self.unitsVals[3*self.M]=val
        return self.unitsVals[3*self.M]
    def BetaGrad(self,val=None):
        if val is not None:
            self.unitsGrads[3*self.M]=val
        return self.unitsGrads[3*self.M]
    def GVal(self,val=None) :
        return self.unitsVals[3*self.M+1]
    def GGrad(self,val=None) :
        if val is not None:
            self.unitsGrads[3*self.M+1]=val
        return self.unitsGrads[3*self.M+1]
    def SVal(self,val=None) :
        return self.unitsVals[3*self.M+2]
    def SGrad(self,val=None) :
        if val is not None:
            self.unitsGrads[3*self.M+2]=val
        return self.unitsGrads[3*self.M+2]
    def GammaVal(self,val=None) :
        if val is not None:
            self.unitsVals[3*self.M+3]=val
        return self.unitsVals[3*self.M+3]
    def GammaGrad(self,val=None) :
        if val is not None:
            self.unitsGrads[3*self.M+3]=val
        return self.unitsGrads[3*self.M+3]

    def headUnitsLen(self):
        return 3*self.M + 4
   

class NTM(object):
    def __init__(self,Controller=None,memOp=None):
        self.Controller=Controller
        self.memOp=memOp
        return
    def backward(self):
        self.memOp.Backward()
        #print self.Controller.heads[0].unitsGrads
        self.Controller.Backward()

def NewNTM(old, x):
   NewController = copy.deepcopy(old.Controller)
   m = NTM(Controller=NewController.Forward(old.memOp.R, x,old.Controller))
   for (i, h) in enumerate(m.Controller.heads):
       h.Wtm1 = old.memOp.W[i]
       #print h.Wtm1
   #print m.Controller.heads[0].Wtm1
   m.memOp = memOp()
   m.memOp.newMemOp(m.Controller.heads, old.memOp.WM)
   #print m.memOp.R[0].TopVal
   #print m.memOp.WM.TopVal
   return m
  
def ForwardBackward(c,in_val,out):
    empty, reads, cas = MakeEmptyNTM(c)
    machines = []
    for t in xrange(len(in_val)):
        #print t
        if t == 0:
            machines.append(NewNTM(empty, in_val[0]))
        else:
            machines.append(NewNTM(machines[t-1], in_val[t]))
        #print machines[t].memOp.WM.TopVal
    for t in range(len(in_val)-1,-1,-1):
        #print t
        m = machines[t]
        out.Model(t, Controller=m.Controller)
        #print m.Controller.yVal
        #print m.Controller.yGrad
        m.backward()
    for (i, read) in enumerate(reads):
        reads[i].Backward()
        cas[i].TopGrad += reads[i].W.TopGrad
        #print cas[i].TopGrad
        cas[i].Backward()
    return machines
  
def MakeEmptyNTM(c):
    wtm1s = []
    reads = []
    cas = []
    for i in xrange(c.NumHeads()):
        wc = contentAddressing()
        wc.newContentAddressing(c.Wtm1BiasV(i))
        #print wc.TopVal
        cas.append(wc)
        W_temp = refocus(wc.TopVal)
        #print W_temp.TopVal
        wtm1s.append(W_temp)
        R_temp=memRead()
        R_temp.newMemRead(W_temp, c.Mtm1BiasV())
        #print R_temp.TopVal
        reads.append(R_temp)
    memOp_temp = memOp(W=wtm1s, R=reads, WM=c.Mtm1BiasV())
    empty = NTM(Controller=c,memOp=memOp_temp)
    return empty, reads, cas

def Predictions(machines ):
    pdts = []
    for t in xrange(len(machines)):
        y = machines[t].Controller.Y()
        pdts.append(y)
    return pdts


class LogisticModel(object):
    def __init__(self,Y=None):
        self.Y=Y
    def Model(self,t,Controller=None,yHs_list=None):
        ys = self.Y[t]
        #print Controller.yVal
        if Controller is not None:
            uVal = Sigmoid(Controller.yVal)
            #print uVal
            Controller.Set_yVal(uVal)
            Controller.Set_yGrad(uVal - ys)
        elif yHs_list is not None:
            uVal = Sigmoid(yHs_list[0])
            #print uVal
            yHs_list[0] = uVal
            yHs_list[1] = uVal - ys
    def Loss(self,output):
        l = 0
        for (t,yh) in enumerate(output):
            #print yh
            for i in xrange(yh.shape[1]):
                #print yh[0]
                p = yh[0,i]
                #print self.Y[t]
                y = self.Y[t][i]
                l += y*numpy.log(p) + (1-y)*numpy.log(1-p)
        #print l
        return -l

class RMSProp(object):
    def __init__(self):
        self.i=0
    def NewRMSProp(self,c):
        self.C = c
        self.N = numpy.zeros(c.NumWeights())
        self.G = numpy.zeros(c.NumWeights())
        self.D = numpy.zeros(c.NumWeights())
    def Train(self,x , y , a, b, c, d ):
        machines = ForwardBackward(self.C, x, y)
        self.i=0
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.C.Weights_Func(self.update)
        return machines
    def update(self,Val,Grad):
        rN = self.a*self.N[self.i] + (1-self.a)*Grad*Grad
        self.N[self.i] = rN

        rG = self.a*self.G[self.i] + (1-self.a)*Grad
        self.G[self.i] = rG

        rD = self.b*self.D[self.i] - self.c*Grad/numpy.sqrt(rN-rG*rG+self.d)
        self.D[self.i] = rD

        self.i += 1
        return Val + rD

class MultinomialModel(object):
    def __init__(self,Y=None):
        self.Y=Y
    def Model(self,t,Controller=None,yHs_list=None):
        k = self.Y[t]
        #print k
        #print Controller.yVal
        if Controller is not None:
            #print Controller.yVal
            y_sum = numpy.sum(numpy.exp(Controller.yVal),axis=1)
            #print y_sum
            uVal = numpy.exp(Controller.yVal)/y_sum
            #print uVal
            Controller.Set_yVal(uVal)
            #print uVal
            uGrad = numpy.zeros_like(uVal)
            for i in xrange(uVal.shape[1]):
                #print "i:",i
                #print "k:",k
                #print "uVal:",uVal[0,i]
                uGrad[0,i] = uVal[0,i] - delta(i,k)
                #print "uVal:",uVal[0,i]
            #print uVal
            Controller.Set_yGrad(uGrad)
        elif yHs_list is not None:
            y_sum = numpy.sum(numpy.exp(yHs_list[0]),axis=1)
            #print y_sum
            uVal = numpy.exp(yHs_list[0])/y_sum
            #print uVal
            yHs_list[0] = uVal
            uGrad = numpy.zeros_like(uVal)
            for i in xrange(uVal.shape[1]):
                uGrad[0,i] = uVal[0,i] - delta(i,k)
            #print uVal
            yHs_list[1] = uGrad
    def Loss(self,output):
        l = 0
        #print output
        for (t,yh) in enumerate(output):
            #print yh
            #print self.Y[t]
            #print yh[0,self.Y[t]]
            l += numpy.log(yh[0,self.Y[t]])
        #print l
        return -l

class LinearModel(object):
    def __init__(self,Y=None):
        self.Y=Y
    def Model(self,t,Controller=None,yHs_list=None):
        ys = self.Y[t]
        #print ys
        #print Controller.yVal
        if Controller is not None:
            uVal = Controller.yVal
            #print uVal
            Controller.Set_yVal(uVal)
            Controller.Set_yGrad(uVal - ys)
            #print uVal - ys
        elif yHs_list is not None:
            uVal = yHs_list[0]
            #print uVal
            yHs_list[0] = uVal
            yHs_list[1] = uVal - ys
    def Loss(self,output):
        l = 0
        for (t,yh) in enumerate(output):
            #print yh
            for i in xrange(yh.shape[1]):
                #print yh[0]
                p = yh[0,i]
                #print self.Y[t]
                y = self.Y[t][i]
                l += (p - y)*(p - y)/2
        #print l
        return l
