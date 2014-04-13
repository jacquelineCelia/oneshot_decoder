/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: model.h   				                                *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "cluster.h"
#include "counter.h"
#include "datum.h"
#include <vector>
#include <map>

using namespace std;

class Model {
 public:
  Model(Config*);
  Model(Model& rhs);
  Model& operator= (Model& rhs);
  vector<Cluster*>& clusters() {return _clusters;}
  Config* config() const {return _config;}
  int weak_limit() const {return _weak_limit;}
  float* A(int i) {return _A[i];}
  float* pi() {return _pi;}
  float** A() {return _A;}
  float* beta() {return _beta;}
  void AddSilenceCluster(Cluster* cluster); 
  void Initialize();
  void Save(const string&);
  void LoadSnapshot(const string&);
  void PreCompute(float**, int);
  void ModifyPi();
  ~Model();
 private:
  vector<Cluster*> _clusters;
  float** _A;
  float* _pi;
  float* _beta;
  Config* _config;
  int _weak_limit;
};

#endif
