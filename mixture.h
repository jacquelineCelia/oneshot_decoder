/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: mixture.h   				                                *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef MIXTURE_H
#define MIXTURE_H

#include "config.h"

class Mixture {
 public:
  Mixture(Config*);
  Mixture(const Mixture&);
  Mixture& operator= (const Mixture&);
  Mixture& operator+= (Mixture&);
  void set_det();
  void set_det(float det); 
  void set_mean(vector<float> mean); 
  void set_pre(vector<float> pre); 
  void Plus(float*);
  void Minus(float*);
  void PreCompute(float**, int);
  float likelihood(float*);
  float likelihood(int);
  float det() const {return _det;}
  int dim() const {return _dim;}
  vector<float> mean() const {return _mean;}
  vector<float> pre() const {return _pre;}
  vector<float> likelihood() const {return _likelihood;}
  Config* config() const {return _config;}
  ~Mixture();
 private:
  Config* _config;
  int _dim;
  float _det;
  vector<float> _mean;
  vector<float> _pre;
  vector<float> _likelihood;
};

#endif
