#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy
import ntm
from math_utils import *

class similarityCircuit(object):
    def __init__(self):
        return
    def newSimilarityCircuit(self,uVal, uGrad, vVal, vGrad):
        self.UVal=uVal
        self.UGrad=uGrad
        self.VVal=vVal
        self.VGrad=vGrad
        #self.VGrad[0][0]=1
        self.UV = numpy.dot(self.UVal, self.VVal.T)
        #print self.UVal
        #print self.VVal
        #print self.UV
        self.Unorm = numpy.linalg.norm(self.UVal)
        self.Vnorm = numpy.linalg.norm(self.VVal)
        self.TopVal = self.UV / (self.Unorm * self.Vnorm)
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def Backward(self):
        uvuu = self.UV / (self.Unorm * self.Unorm)
        uvvv = self.UV / (self.Vnorm * self.Vnorm)
        uvg = self.TopGrad / (self.Unorm * self.Vnorm)
        self.UGrad( val= self.UGrad() + (self.VVal - self.UVal*uvuu) * uvg)
        self.VGrad += (self.UVal - self.VVal*uvvv) * uvg
        #print self.VGrad

class similarityCircuitOne(similarityCircuit):
    def __init__(self):
        return
    def newSimilarityCircuit(self,uVal, uGrad, vVal, vGrad):
        self.UVal=uVal
        self.UGrad=uGrad
        self.VVal=vVal
        self.VGrad=vGrad
        self.TopVal = -(uVal-vVal)*(uVal-vVal)
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def Backward(self):
        uvg = self.TopGrad 
        self.UGrad( val= self.UGrad() - 2*(self.UVal-self.VVal) * uvg)
        self.VGrad += 2*(self.UVal - self.VVal) * uvg
        #print self.VGrad

class betaSimilarity(object):
    def __init__(self):
        self.TopVal=numpy.zeros((1,1))
        self.TopGrad=numpy.zeros(self.TopVal.shape)
        return
    def newBetaSimilarity(self,betaVal , betaGrad , s ):
        self.BetaVal=betaVal
        self.BetaGrad=betaGrad
        self.S=s
        self.b=numpy.exp(betaVal)
        self.TopVal = self.b * s.TopVal
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def Backward(self):
        self.BetaGrad(val= self.BetaGrad()+self.S.TopVal * self.b * self.TopGrad)
        self.S.TopGrad += self.b * self.TopGrad
        #print self.S.TopGrad.shape

class contentAddressing(object):
    def __init__(self):
        return
    def newContentAddressing(self,units):
        self.Units=units
        self.TopVal=numpy.zeros((1,len(self.Units)))
        self.TopGrad=numpy.zeros((1,len(self.Units)))
        maxVal = self.Units[0].TopVal[0][0]
        for u in self.Units :
            #print maxVal
            #print u.TopVal
            maxVal =max(maxVal,u.TopVal[0][0])
        sum = 0
        for (i,u) in enumerate(self.Units):
            w = numpy.exp(u.TopVal[0][0] - maxVal)
            self.TopVal[0][i] = w
            sum += w
        self.TopVal = self.TopVal/sum
    def Backward(self):
        gv  = 0
        gv = sum(self.TopGrad[0]*self.TopVal[0])
        for (i,u) in enumerate(self.Units):
            u.TopGrad[0][0] += (self.TopGrad[0][i] - gv) * self.TopVal[0][i]
            #print u.TopGrad[0][0]

class gatedWeighting(object):
    def __init__(self):
        return
    def newGatedWeighting(self,gVal , gGrad , wc , wtm1  ):
        self.GVal = gVal
        self.GGrad = gGrad
        self.WC= wc
        self.Wtm1=wtm1
        gt = Sigmoid(gVal)
        self.TopVal = gt*wc.TopVal + (1-gt)*wtm1.TopVal
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def Backward(self):
        gt = Sigmoid(self.GVal)
        grad  = sum(((self.WC.TopVal-self.Wtm1.TopVal)*self.TopGrad)[0])
        self.GGrad(val=self.GGrad() + grad * gt * (1 - gt))
        #print self.GGrad()
        self.WC.TopGrad += gt * self.TopGrad
        #print self.WC.TopGrad
        self.Wtm1.TopGrad += (1 - gt) * self.TopGrad
        #print self.Wtm1.TopGrad

class shiftedWeighting(object):
    def __init__(self):
        return
    def newShiftedWeighting(self,sVal , sGrad , wg ):
        self.SVal= sVal
        self.SGrad=sGrad
        self.WG=wg
        self.TopVal=numpy.zeros(wg.TopVal.shape)
        self.TopGrad=numpy.zeros(self.TopVal.shape)
        n = len(self.WG.TopVal[0])
        shift = (2*Sigmoid(sVal) - 1)
        self.Z = numpy.mod(shift+numpy.float32(n), numpy.float32(n))
        simj = 1 - (self.Z - numpy.floor(self.Z))
        for i in  xrange(len(self.TopVal[0])):
            imj = (i + int(self.Z)) % n
            self.TopVal[0][i] = self.WG.TopVal[0][imj]*simj + self.WG.TopVal[0][(imj+1)%n]*(1-simj)
    def Backward(self):
        grad  = 0
        n = len(self.WG.TopVal[0])
        for i in  xrange(len(self.TopVal[0])):
            imj = (i + int(self.Z)) % n
            grad += (-self.WG.TopVal[0][imj]+ self.WG.TopVal[0][(imj+1)%n])*self.TopGrad[0][i]
        sig = Sigmoid(self.SVal)
        grad = grad * 2 * sig * (1 - sig)
        self.SGrad(val=self.SGrad() + grad)
        #print self.SGrad()
        simj = 1 - (self.Z - numpy.floor(self.Z))
        for i in  xrange(len(self.TopVal[0])):
            j = (i - int(self.Z)) % n
            self.WG.TopGrad[0][i] = self.TopGrad[0][j]*simj + self.TopGrad[0][(j-1+n)%n]*(1-simj)
        #print self.WG.TopGrad

class refocus(object):
    def __init__(self, TopVal=None):
        self.TopVal=TopVal
        if TopVal is not None:
            self.TopGrad=numpy.zeros(self.TopVal.shape)
    def newRefocus(self,gammaVal, gammaGrad, sw):
        self.GammaVal =gammaVal
        self.GammaGrad=gammaGrad
        #self.GammaGrad(val=1)
        self.SW=sw
        self.g=numpy.log(numpy.exp(gammaVal)+1) + 1
        self.TopVal = pow(sw.TopVal,self.g)
        sum = numpy.sum(self.TopVal)
        self.TopVal = self.TopVal /sum
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def backwardSW(self):
        topGV  = sum(self.TopGrad[0]*self.TopVal[0])
        self.SW.TopGrad += (self.TopGrad - topGV)*self.g/self.SW.TopVal*self.TopVal
        #print self.g
        #print self.SW.TopGrad
    def backwardGamma(self):
        lns=numpy.log(self.SW.TopVal)
        pow_val=pow(self.SW.TopVal,self.g)
        lnexp=sum(lns[0]*pow_val[0])
        s=sum(pow_val[0])
        lnexps = lnexp / s
        #print lnexps
        grad = sum((self.TopGrad * (self.TopVal * (lns - lnexps)))[0])
        grad = grad / (1 + numpy.exp(-self.GammaVal))
        self.GammaGrad(val=self.GammaGrad() + grad)
        #print self.GammaGrad()
    def Backward(self):
        self.backwardSW()
        self.backwardGamma()

class memRead(object):
    def __init__(self):
        return
    def newMemRead(self,w,memory):
        self.W=w
        self.Memory=memory
        self.TopVal=numpy.dot(self.W.TopVal,self.Memory.TopVal)
        self.TopGrad=numpy.zeros(self.TopVal.shape)
    def Backward(self):
        #print self.TopGrad
        #print self.Memory.TopVal
        self.W.TopGrad += numpy.dot(self.TopGrad,self.Memory.TopVal.T)
        #print self.W.TopVal
        #print self.W.TopGrad
        self.Memory.TopGrad += numpy.dot(self.W.TopVal.T,self.TopGrad)
        #print self.Memory.TopGrad

class writtenMemory(object):
    def __init__(self, N=0,TopVal=None,TopGrad=None):
        self.TopVal=TopVal
        self.N=N
        self.TopGrad=TopGrad
    def newWrittenMemory(self,ws , heads , mtm1 ):
        n = mtm1.N
        m = mtm1.TopVal.shape[1]
        #print n,m
        self.Ws=ws
        self.Heads=heads
        self.Mtm1=mtm1
        self.N=mtm1.N
        self.TopVal=numpy.zeros(mtm1.TopVal.shape)
        self.TopGrad=numpy.zeros(mtm1.TopVal.shape)
        self.erase=numpy.zeros((len(heads), m))
        self.add=numpy.zeros((len(heads), m))
        self.erasures=numpy.zeros(mtm1.TopVal.shape)
        #print self.erasures
        for (i, h) in enumerate(heads):
            #print self.erase[i]
            #print h.EraseVal()
            self.erase[i] = Sigmoid(h.EraseVal()[0])
            self.add[i] = Sigmoid(h.AddVal()[0])
        #for (k, weight) in  enumerate(self.Ws):
        #    self.erasure[k] = 1-numpy.dot(weight.TopVal,self.erase) #[1 -wt(i)e(t)]
        #    self.add[k] = numpy.dot(weight.TopVal,self.add)
        #self.TopVal = numpy.dot(self.erasure,mtm1.TopVal) + self.add
        for (i, mtm1Row) in enumerate(self.Mtm1.TopVal):
            erasure = self.erasures[i]
            for (j, mtm1) in enumerate(mtm1Row):
                e = 1
                adds = 0
                for (k, weights) in enumerate(self.Ws):
                    #print weights.TopVal
                    e = e * (1 - weights.TopVal[0][i]*self.erase[k][j])
                    adds += weights.TopVal[0][i] * self.add[k][j]
                #print e
                erasure[j] = e
                self.TopVal[i][j] += e*mtm1 + adds
    def backwardWErase(self):
        grad = 0
        #print self.Heads[1].EraseGrad()
        for (i, weights) in enumerate(self.Ws):
            #print self.Heads[i].EraseGrad()
            erase = self.erase[i]
            add = self.add[i]
            ws = self.Ws[i]
            for (j, topRow) in enumerate(self.TopGrad):
                mtm1Row = self.Mtm1.TopVal[j]
                erasure = self.erasures[j]
                wsj = ws.TopVal[0][j]
                grad = 0
                for (k, mtm1) in enumerate(mtm1Row):
                    mtilt = mtm1
                    e = erase[k]
                    #print wsj
                    mwe = 1 - wsj*e
                    #print mwe
                    if numpy.abs(mwe) > 1e-6 :
                        mtilt = mtilt * erasure[k] / mwe
                    else :
                        for (q, ws) in enumerate(wm.Ws):
                            if q == i:
                                continue
                            mtilt = mtilt * (1 - ws.TopVal[0][j]*wm.erase[q][k])
                    grad += (mtilt*(-e) + add[k]) * topRow[k]
                    #hErase[k].Grad += topRow[k].Grad * mtilt * (-wsj)
                    #print i
                    #print self.Heads[i].EraseGrad(k=k)
                    self.Heads[i].EraseGrad(k=k,val=self.Heads[i].EraseGrad(k=k)+topRow[k] * mtilt * (-wsj))
                weights.TopGrad[0][j] += grad
            #print self.Heads[i].EraseGrad()
            for (j, e) in enumerate(erase) :
                #hErase[j].Grad *= e * (1 - e)
                self.Heads[i].EraseGrad(k=j,val=self.Heads[i].EraseGrad(k=j)*e * (1 - e))
            #print self.Heads[i].EraseGrad()
    def backwardAdd(self):
        grad=0
        for (k, h) in enumerate(self.Heads):
            add = self.add[k]
            ws = self.Ws[k]
            hAdd = h.AddVal()[0]
            for i in xrange(len(hAdd)):
                grad = 0
                for (j, toprow) in enumerate(self.TopGrad):
                    #print ws.TopVal[j]
                    grad += toprow[i] * ws.TopVal[0][j]
                a = add[i]
                h.AddGrad(k=i,val=h.AddGrad(k=i) + grad * a * (1 - a))
            #print h.AddGrad()
    def backwardMtm1(self):
        n = self.TopVal.shape[0]
        m = self.TopVal.shape[1]
        #print self.erase
        for i in xrange(n):
            for j in xrange(m):
                grad = 1
                for (q, ws) in enumerate(self.Ws):
                    #print ws.TopVal[0][i]
                    #print q,j
                    #print self.erase[q][j]
                    grad = grad * (1 - ws.TopVal[0][i]*self.erase[q][j])
                self.Mtm1.TopGrad[i][j] += grad * self.TopGrad[i][j]
        #print self.Mtm1.TopGrad
    def Backward(self):
        self.backwardWErase()
        self.backwardAdd()
        self.backwardMtm1()

class memOp(object):
    def __init__(self,W=None,R=None,WM=None):
        self.W = W
        self.R = R
        self.WM = WM
        return
    def newMemOp(self,heads,mtm1):
        self.R=[]
        self.W=[]
        for (wi, h) in enumerate(heads):
            ss=[]
            for i in xrange(mtm1.N):
                #m = mtm1.TopVal.shape[1]
                #print "h.KVal().shape",h.KVal().shape
                if h.KVal().shape[0] == 1 and h.KVal().shape[1] == 1:
                    s = similarityCircuitOne()
                else:
                    s = similarityCircuit()
                
                #print mtm1.TopVal
                #print mtm1.TopVal[i]
                #print mtm1.TopVal[i:(i+1)]
                s.newSimilarityCircuit(h.KVal(), h.KGrad, mtm1.TopVal[i:(i+1)], mtm1.TopGrad[i:(i+1)])
                #print s.TopVal
                #print mtm1.TopGrad[i:(i+1)]
                ss_temp =betaSimilarity()
                ss_temp.newBetaSimilarity(h.BetaVal(), h.BetaGrad, s)
                #print ss_temp.TopVal
                ss.append(ss_temp)
            wc = contentAddressing()
            wc.newContentAddressing(ss)
            #print wc.TopVal
            wg=gatedWeighting()
            wg.newGatedWeighting(h.GVal(), h.GGrad, wc, h.Wtm1)
            #print wg.TopVal
            ws =shiftedWeighting()
            ws.newShiftedWeighting(h.SVal(), h.SGrad, wg)
            #print ws.TopVal
            W_temp = refocus()
            W_temp.newRefocus(h.GammaVal(), h.GammaGrad, ws)
            #print h.GammaGrad()
            #print W_temp.TopVal
            self.W.append(W_temp)
            R_temp=memRead()
            R_temp.newMemRead(W_temp, mtm1)
            #print R_temp.TopVal
            self.R.append(R_temp)
        self.WM = writtenMemory()
        self.WM.newWrittenMemory(self.W, heads, mtm1)
        #print self.WM.TopVal
    def Backward(self):
        for r in self.R :
            r.Backward()
        self.WM.Backward()
        for rf in self.WM.Ws :
            rf.Backward()
            rf.SW.Backward()
            rf.SW.WG.Backward()
            rf.SW.WG.WC.Backward()
            for bs in rf.SW.WG.WC.Units:
                bs.Backward()
                bs.S.Backward()