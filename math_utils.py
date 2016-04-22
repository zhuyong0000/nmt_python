#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import numpy

machineEpsilon     = 2.2e-16
machineEpsilonSqrt = 1e-8 # math.Sqrt(machineEpsilon)

def Sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))
   

def cosineSimilarity(u, v ):
    uvsum = sum(u * v)
    usum = sum(u * u)
    vsum = sum(v * v)
    return uvsum / numpy.sqrt(usum*vsum)

def delta(a, b ):
    if a == b :
        return 1
    return 0
