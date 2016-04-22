#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy
from ntm import Head
from addressing import writtenMemory,refocus,memOp ,betaSimilarity
from math_utils import *

class controller1(object):
    def __init__(self):
        return
    def NewEmptyController1(self,xSize, ySize, h1Size, numHeads, n, m ):
        #print xSize, ySize, h1Size, numHeads, n, m
        self.h=Head()
        self.h.NewHead(m)
        headUnitsSize = self.h.unitsVals.shape[0]
        self.mtm1 = writtenMemory(N=n,TopVal=numpy.zeros((n,m)),TopGrad=numpy.zeros((n,m)))
        self.Wh1rVal = numpy.zeros((h1Size, numHeads, m))
        #print self.Wh1rVal
        self.Wh1xVal = numpy.zeros((h1Size, xSize))
        self.Wh1bVal = numpy.zeros((1,h1Size))
        self.Wyh1Val = numpy.zeros((ySize, h1Size+1))
        self.Wuh1Val = numpy.zeros((numHeads, headUnitsSize, h1Size+1))
        self.Wh1rGrad = numpy.zeros(self.Wh1rVal.shape)
        self.Wh1xGrad = numpy.zeros(self.Wh1xVal.shape)
        self.Wh1bGrad = numpy.zeros(self.Wh1bVal.shape)
        self.Wyh1Grad = numpy.zeros(self.Wyh1Val.shape)
        self.Wuh1Grad = numpy.zeros(self.Wuh1Val.shape)
        #print self.Wuh1Val.shape
        self.wtm1s =[]
        for i in xrange(numHeads):
            temp_list = []
            for j in xrange(n) :
                temp_similay = betaSimilarity()
                temp_list.append(temp_similay)
            self.wtm1s.append(temp_list)
        self.numWeights = numHeads*n + n*m + h1Size*numHeads*m + h1Size*xSize + h1Size + ySize*(h1Size+1) + numHeads*headUnitsSize*(h1Size+1)
    def Forward(self,reads, x,old):
        self.Wh1rGrad = old.Wh1rGrad
        self.Wh1xGrad = old.Wh1xGrad
        self.Wh1bGrad = old.Wh1bGrad
        self.Wyh1Grad = old.Wyh1Grad
        self.Wuh1Grad = old.Wuh1Grad
        self.Reads=reads
        self.X= x
        #print self.Wh1rVal
        Wh1rdotread = numpy.zeros(self.Wh1bVal.shape)
        for i in xrange(self.Wh1rVal.shape[0]):
            v=0
            for (j, read) in enumerate(reads):
                #print read.TopVal
                v += numpy.dot(self.Wh1rVal[i][j],read.TopVal.T)
                #print v
            Wh1rdotread[0][i] = v
        self.H1Val = Sigmoid(numpy.dot(self.Wh1xVal,x)+Wh1rdotread+self.Wh1bVal)
        #print self.H1Val
        #print self.Wyh1Val
        #print self.Wyh1Val[:,0:self.H1Val.shape[1]]
        self.yVal = numpy.dot(self.H1Val,self.Wyh1Val[:,0:self.H1Val.shape[1]].T) + self.Wyh1Val[:,self.H1Val.shape[1]:].T
        #print self.yVal
        #print self.Wuh1Val
        memoryM = reads[0].TopVal.shape[1]
        self.heads = []
        for i in xrange(self.Wuh1Val.shape[0]):
            head_temp = Head()
            head_temp.NewHead(memoryM)
            Wuh1ValdotH1 =  numpy.dot(self.H1Val,self.Wuh1Val[i,:,0:self.H1Val.shape[1]].T) + self.Wuh1Val[i,:,self.H1Val.shape[1]:].T
            #print Wuh1ValdotH1
            #print Wuh1ValdotH1[0,0]
            for j in xrange(head_temp.headUnitsLen()):
                head_temp.unitsVals[j] = Wuh1ValdotH1[0,j]
            self.heads.append(head_temp)
            #print self.heads[i].unitsVals
        self.yGrad = numpy.zeros(self.yVal.shape)
        self.H1Grad = numpy.zeros(self.H1Val.shape)
        return self
    def Backward(self):
        #print self.yGrad
        #print self.H1Val
        #print self.Wyh1Grad
        #print self.Wyh1Val[:,0:self.H1Val.shape[1]]
        self.H1Grad = self.H1Grad + numpy.dot(self.yGrad,self.Wyh1Val[:,0:self.H1Val.shape[1]])
        #print self.H1Grad
        for (j, head) in enumerate(self.heads) :
            wuh1j = self.Wuh1Val[j]
            #print head.unitsGrads
            self.H1Grad = self.H1Grad + numpy.dot(head.unitsGrads,wuh1j[:,0:self.H1Val.shape[1]])
        #print self.H1Grad
        self.Wyh1Grad[:,0:self.H1Val.shape[1]] += numpy.dot(self.H1Val.T,self.yGrad).T
        self.Wyh1Grad[:,self.H1Val.shape[1]:] += self.yGrad.T
        #print self.Wyh1Grad
        #self.Wuh1Grad = numpy.zeros(self.Wuh1Val.shape)
        for (j, head) in enumerate(self.heads) :
            wuh1j = self.Wuh1Grad[j]
            headGrads = head.unitsGrads.reshape(1,head.unitsGrads.shape[0])
            wuh1j[:,0:self.H1Val.shape[1]] +=  numpy.dot(self.H1Val.T,headGrads).T
            wuh1j[:,self.H1Val.shape[1]:] += headGrads.T
        #print self.Wuh1Grad
        #print self.H1Grad.shape
        h1Grads =  numpy.multiply(numpy.multiply(self.H1Grad,self.H1Val),(1 - self.H1Val))
        #print h1Grads
        #print self.Wh1rVal
        for (j, read) in enumerate(self.Reads):
            for i in xrange(self.Wh1rVal.shape[0]):
                #print self.Wh1rVal[i][j]
                #print h1Grads[0,j]
                read.TopGrad[0] = read.TopGrad[0] + self.Wh1rVal[i][j]*h1Grads[0,i]
        #print self.Reads[1].TopGrad
        #self.Wh1rGrad = numpy.zeros(self.Wh1rVal.shape)
        for (j, read) in enumerate(self.Reads):
            for i in xrange(self.Wh1rVal.shape[0]):
                self.Wh1rGrad[i][j]  = self.Wh1rGrad[i][j] + read.TopVal[0]*h1Grads[0,i]
        #print self.Wh1rGrad
        #print self.Wh1xVal.shape
        self.Wh1xGrad += numpy.dot(h1Grads.T,self.X.reshape(1,self.X.shape[0]))
        #print self.Wh1xGrad
        self.Wh1bGrad += h1Grads
        #print self.Wh1bGrad
    def Wtm1BiasV(self,k=None):
        if k is not None:
            return self.wtm1s[k]
        else:
            return self.wtm1s
    def Mtm1BiasV(self):
        return self.mtm1
    def Weights(self,weights_list):
        weight_index=0
        for wtm1 in self.wtm1s:
            for  w in wtm1:
                w.TopVal[0][0]=weights_list[weight_index]
                weight_index=weight_index+1
                #print w.TopVal
        for i in xrange(self.mtm1.TopVal.shape[0]):
            for j in xrange(self.mtm1.TopVal.shape[1]) :
                self.mtm1.TopVal[i][j] = weights_list[weight_index]
                weight_index=weight_index+1
        #print self.mtm1.TopVal
        self.Wyh1Val = numpy.matrix(weights_list[weight_index:weight_index+self.Wyh1Val.shape[0]*self.Wyh1Val.shape[1]]).reshape(self.Wyh1Val.shape)
        weight_index=weight_index+self.Wyh1Val.shape[0]*self.Wyh1Val.shape[1]
        #print self.Wyh1Val
        for i in xrange(self.Wuh1Val.shape[0]):
            for j in xrange(self.Wuh1Val.shape[1]) :
                for k in xrange(self.Wuh1Val.shape[2]) :
                    self.Wuh1Val[i][j][k] = weights_list[weight_index]
                    weight_index=weight_index+1
        #print self.Wuh1Val
        for i in xrange(self.Wh1rVal.shape[0]):
            for j in xrange(self.Wh1rVal.shape[1]) :
                for k in xrange(self.Wh1rVal.shape[2]) :
                    self.Wh1rVal[i][j][k] = weights_list[weight_index]
                    weight_index=weight_index+1
        #print self.Wh1rVal
        self.Wh1xVal = numpy.matrix(weights_list[weight_index:weight_index+self.Wh1xVal.shape[0]*self.Wh1xVal.shape[1]]).reshape(self.Wh1xVal.shape)
        weight_index=weight_index+self.Wh1xVal.shape[0]*self.Wh1xVal.shape[1]
        #print self.Wh1xVal
        self.Wh1bVal = numpy.matrix(weights_list[weight_index:weight_index+self.Wh1bVal.shape[1]])
        weight_index=weight_index+self.Wh1bVal.shape[1]
        #print self.Wh1bVal
    def Weights_Func(self,func):
        weight_index=0
        for wtm1 in self.wtm1s:
            for  w in wtm1:
                w.TopVal[0][0]=func(w.TopVal[0][0],w.TopGrad[0][0])
                #print w.TopVal
        for i in xrange(self.mtm1.TopVal.shape[0]):
            for j in xrange(self.mtm1.TopVal.shape[1]) :
                self.mtm1.TopVal[i][j] = func(self.mtm1.TopVal[i][j],self.mtm1.TopGrad[i][j])
        #print self.mtm1.TopVal
        #print self.Wyh1Val.shape
        for i in xrange(self.Wyh1Val.shape[0]):
            for j in xrange(self.Wyh1Val.shape[1]) :
                #print i,j
                #print self.Wyh1Val[i,j]
                self.Wyh1Val[i,j] = func(self.Wyh1Val[i,j],self.Wyh1Grad[i,j])
        #print self.Wyh1Val
        for i in xrange(self.Wuh1Val.shape[0]):
            for j in xrange(self.Wuh1Val.shape[1]) :
                for k in xrange(self.Wuh1Val.shape[2]) :
                    self.Wuh1Val[i][j][k] = func(self.Wuh1Val[i][j][k],self.Wuh1Grad[i][j][k])
        #print self.Wuh1Val
        for i in xrange(self.Wh1rVal.shape[0]):
            for j in xrange(self.Wh1rVal.shape[1]) :
                for k in xrange(self.Wh1rVal.shape[2]) :
                    self.Wh1rVal[i][j][k] = func(self.Wh1rVal[i][j][k],self.Wh1rGrad[i][j][k])
        #print self.Wh1rVal
        for i in xrange(self.Wh1xVal.shape[0]):
            for j in xrange(self.Wh1xVal.shape[1]) :
                self.Wh1xVal[i,j] = func(self.Wh1xVal[i,j],self.Wh1xGrad[i,j])
        #print self.Wh1xVal
        for i in xrange(self.Wh1bVal.shape[0]):
            for j in xrange(self.Wh1bVal.shape[1]) :
                self.Wh1bVal[i,j] = func(self.Wh1bVal[i,j],self.Wh1bGrad[i,j])
        #print self.Wh1bVal
    def get_Weights(self):
        weights_list = []
        weights_grad_list = []
        weight_index=0
        for wtm1 in self.wtm1s:
            for  w in wtm1:
                weights_list.append(w.TopVal[0][0])
                weights_grad_list.append(w.TopGrad[0][0])
                weight_index=weight_index+1
                #print w.TopVal
        for i in xrange(self.mtm1.TopVal.shape[0]):
            for j in xrange(self.mtm1.TopVal.shape[1]) :
                weights_list.append(self.mtm1.TopVal[i][j])
                weights_grad_list.append(self.mtm1.TopGrad[i][j])
                weight_index=weight_index+1
        #print self.mtm1.TopVal
        for i in xrange(self.Wyh1Val.shape[0]):
            for j in xrange(self.Wyh1Val.shape[1]) :
                weights_list.append(self.Wyh1Val[i,j])
                weights_grad_list.append(self.Wyh1Grad[i,j])
                weight_index=weight_index+1
        #print self.Wyh1Val
        for i in xrange(self.Wuh1Val.shape[0]):
            for j in xrange(self.Wuh1Val.shape[1]) :
                for k in xrange(self.Wuh1Val.shape[2]) :
                    weights_list.append(self.Wuh1Val[i][j][k])
                    weights_grad_list.append(self.Wuh1Grad[i][j][k])
                    weight_index=weight_index+1
        #print self.Wuh1Val
        for i in xrange(self.Wh1rVal.shape[0]):
            for j in xrange(self.Wh1rVal.shape[1]) :
                for k in xrange(self.Wh1rVal.shape[2]) :
                    weights_list.append(self.Wh1rVal[i][j][k])
                    weights_grad_list.append(self.Wh1rGrad[i][j][k])
                    weight_index=weight_index+1
        #print self.Wh1rVal
        for i in xrange(self.Wh1xVal.shape[0]):
            for j in xrange(self.Wh1xVal.shape[1]) :
                weights_list.append(self.Wh1xVal[i,j])
                weights_grad_list.append(self.Wh1xGrad[i,j])
                weight_index=weight_index+1
        #print self.Wh1xVal
        for j in xrange(self.Wh1bVal.shape[1]) :
            weights_list.append(self.Wh1bVal[0,j])
            weights_grad_list.append(self.Wh1bGrad[0,j])
            weight_index=weight_index+1
        return weights_list,weights_grad_list
    def NumWeights(self):
        return self.numWeights
    def NumHeads(self):
       return self.Wuh1Val.shape[0]
    def MemoryN(self):
       return self.mtm1.N
    def MemoryM(self):
       return self.Wh1rVal.shape[2]
    def Set_yGrad(self,yGrad):
        self.yGrad = yGrad
    def Set_yVal(self,value):
        self.yVal = value
    def Y(self):
        return self.yVal