#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy

import ntm
from ntm import Head
from addressing import writtenMemory,refocus,memOp ,betaSimilarity
from math_utils import *

logger = logging.getLogger(__name__)

outputGradient    = 1.234
w11OutputGradient = 0.987

#zy test
memory_list = numpy.array([[0.604660, 0.940509],[0.664560, 0.437714],[0.424637,0.686823]])
randomRefocus_list = numpy.array([[0.065637, 0.156519, 0.096970],[0.218553, 0.203187,0.360871]])
heads_list = numpy.array([[0.300912, 0.515213, 0.813640,0.214264,0.380657,0.318058,0.468890,0.283034,0.293102,0.679085],[0.570673, 0.862491,0.293114,0.297083,0.752573,0.206583,0.865335,0.696719,0.523820,0.028303]])
#zy test end

def TestCircuit():
    n = 3
    m = 2
    #memory = writtenMemory(n,numpy.random.rand(n,m),numpy.zeros((n,m)))
    memory = writtenMemory(N=n,TopVal=memory_list,TopGrad=numpy.zeros((n,m)))
    #print memory.Top
    heads = []
    for i in xrange(2):
        head=Head()
        head.NewHead(m)
        hul = head.headUnitsLen()
        head.unitsVals = heads_list[i]#numpy.random.rand(hul)
        head.unitsGrads = numpy.zeros(hul)
        head.Wtm1 = randomRefocus(n,randomRefocus_list[i])
        heads.append(head)
        #print head.unitsVals
        #print head.Wtm1.Top
    heads[0].BetaVal(val=0.137350) 
    heads[0].GammaVal(val=1.9876)
    circuit = memOp()
    circuit.newMemOp(heads, memory)
    for (k, weights) in enumerate(circuit.W):
        weights.TopGrad = weights.TopGrad + outputGradient
        if 0 == k:
            weights.TopGrad[0][0] =  w11OutputGradient
        #print weights.TopGrad
    for (k, weights) in enumerate(circuit.R):
        weights.TopGrad = weights.TopGrad + outputGradient
        #print weights.TopGrad
    circuit.WM.TopGrad = circuit.WM.TopGrad + outputGradient
    #print circuit.WM.TopGrad
    #print circuit.WM.erase
    #print circuit.WM.add
    #print circuit.WM.erasures
    circuit.Backward()
    ax = addressing(heads, memory)
    checkGamma(heads, memory, ax)

def addressing(heads, memory):
    #return addressingLoss(doAddressing(heads, memory))
    weights, reads, newMem=doAddressing(heads, memory)
    return addressingLoss(weights, reads, newMem)

def doAddressing(heads,memory):
    weights = numpy.zeros((len(heads), memory.N))
    #print weights
    for (i, h) in enumerate(heads):
        beta = numpy.exp(h.BetaVal())
        wc = numpy.zeros((1, memory.N))
        for j in xrange(wc.shape[1]):
            wc[0][j] = numpy.exp(beta * cosineSimilarity(h.KVal()[0], memory.TopVal[j]))
        #print wc
        sum = numpy.sum(wc)
        wc = wc / sum
        #print wc # the Similarity between KVal and memory
        g = Sigmoid(h.GVal())
        wc = g*wc + (1-g)*h.Wtm1.TopVal
        #print wc # if g=0 can ignore this time Similarity (wc),only use the last time Similarity (h.Wtm1)
        n = weights.shape[1]
        s = numpy.mod((2*Sigmoid(h.SVal())-1)+n, n)
        #print s
        for j in  xrange(n):
            imj = (j + int(s)) % n
            simj = 1 - (s - numpy.floor(s))
            weights[i][j] = wc[0][imj]*simj + wc[0][(imj+1)%n]*(1-simj)
        #print weights
        gamma = numpy.log(numpy.exp(h.GammaVal())+1) + 1
        weights[i] = pow(weights[i], gamma)
        sum = numpy.sum(weights[i])
        weights[i] = weights[i]/sum
    #print weights
    reads = numpy.dot(weights,memory.TopVal) 
    #print reads
    erase = numpy.zeros((len(heads), memory.TopVal.shape[1]))
    add = numpy.zeros((len(heads), memory.TopVal.shape[1]))
    for k in xrange(len(heads)):
        eraseVec = heads[k].EraseVal()
        erase[k] = Sigmoid(eraseVec)
        addVec = heads[k].AddVal()
        add[k] = Sigmoid(addVec)
    #print erase
    #print add
    newMem =  numpy.zeros(memory.TopVal.shape)
    for i in xrange(newMem.shape[0]):
        for j in xrange(newMem.shape[1]):
            newMem[i][j] = memory.TopVal[i][j]
            for k in xrange(len(heads)):
                newMem[i][j] = newMem[i][j]*(1 - weights[k][i]*erase[k][j]) #notice this is mutiply for every elem not dot
    newMem += numpy.dot(weights.T ,  add)
    #print newMem
    return weights, reads, newMem

def addressingLoss(weights , reads , newMem ):
    res =0
    for i in xrange(weights.shape[0]):
        for j in xrange(weights.shape[1]):
            if i == 0 and j == 0 :
                res += weights[i][j] * w11OutputGradient
            else:
                res += weights[i][j] * outputGradient
    #print res
    res += numpy.sum(reads * outputGradient)
    #print res
    res += numpy.sum(newMem * outputGradient)
    #print res
    return res

def checkGamma(heads ,memory , ax ):
    #print memory.TopVal
    for (k, hd) in  enumerate(heads) :
        #print hd.unitsVals
        x = hd.GammaVal()
        h = machineEpsilonSqrt * max(numpy.abs(x), 1)
        xph = x + h
        #print xph
        #print hd.GammaVal()
        hd.GammaVal(val=xph)
        dx = xph - x
        #print dx
        #print checkGamma
        axph = addressing(heads, memory)
        #print axph
        #print ax
        grad = (axph - ax) / dx
        #print grad
        hd.GammaVal(val=x)
        if numpy.abs(grad-hd.GammaGrad()) > 1e-5 :
            print "wrong gamma gradient expected ",grad, "got ",  hd.GammaGrad()
        else:
            print "OK gamma",k," gradient ",grad," ", hd.GammaGrad()

def randomRefocus(n,list):
    #w = numpy.zeros(n)
    sum = 0
    w = list#numpy.abs(numpy.random.rand(n))
    sum = numpy.sum(w)
    w = w /sum
    #print w.shape
    return refocus(TopVal=w.reshape((1,w.shape[0])))

if __name__ == "__main__":
    TestCircuit()