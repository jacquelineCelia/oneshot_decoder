/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: gmm.h   				                                    *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef GMM_H
#define GMM_H

#include <vector>
#include <fstream>
#include <iostream>

#include "config.h"
#include "mixture.h"
#include "toolkit.h"

using namespace std;

class GMM{
 public:
  GMM(Config*);
  GMM(const GMM&);
  GMM(Config*, int);
  GMM& operator= (const GMM&);
  GMM& operator+= (GMM&);
  vector<Mixture> mixtures() const {return _mixtures;}
  Mixture& mixture(int index) {return _mixtures[index];}
  Config* config() const {return _config;}
  int mix_num() const {return _mix_num;}
  float ComputeLikehood(float*);
  float ComputeLikehood(int);
  vector<float> ComponentLikelihood(float*);
  vector<float> ComponentLikelihood(int);
  vector<float> weight() const {return _weight;}
  void set_mixture(Mixture&, int);
  void set_mixtures(vector<Mixture>& mixtures);
  void set_weight(vector<float> wegith);
  void Minus(float*, int);
  void Plus(float*, int);
  void PreCompute(float**, int);
  void ComputeLikehood(vector<float*>, float*);
  void ComputeLikehood(int, int, float*);
  void Save(ofstream&);
  void Load(ifstream&);
  ~GMM(); 
 private:
  // Weight stored in Log
  vector<float> _weight;
  vector<Mixture> _mixtures;
  int _mix_num;
  ToolKit _toolkit;
  Config* _config;
};

#endif
