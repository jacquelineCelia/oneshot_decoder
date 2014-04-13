/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: decoder.h
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef DECODER_H
#define DECODER_H

#include "config.h"
#include "datum.h"
#include "cluster.h"
#include "model.h"
#include "prob_list.h"

class Decoder {
 public:
  Decoder(Config*);
  void GetMostLikelySegments(Datum*, Model*);
  void ComputeSegProbGivenCluster(Datum*, vector<Cluster*>, float***);
  void MessageBackward(Datum*, float*, float**, float***, \
          ProbList<int> **, ProbList<int>**);
  void DecodeForward(Datum*, ProbList<int>**, ProbList<int>**);
  float ComputeLikelihoodGivenFixedDecoding(Datum*, vector<GMM*>, float**);
  int FindIndexForMaxEntry(vector<float>);
  ~Decoder() {};
 private:
  Config* _config;
}; 

#endif
