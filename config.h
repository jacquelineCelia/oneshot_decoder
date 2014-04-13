/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: config.h 
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>

using namespace std;

class Config {
 public:
  Config();
  ~Config();
  bool Load(string& fn, string& fn_gaussian);
  bool Load(string& fn);
  bool LoadGaussian(string& fn_gaussian);
  bool parallel() {return _parallel;}
  bool precompute() {return _precompute;} 
  int state_num() {return _state_num;}
  int max_duration() {return _max_duration;}
  int cluster_num() {return _cluster_num;}
  int mix_num() {return _mix_num;}
  int dim() {return _dim;}
  int weak_limit() {return _weak_limit;}
  int num_sil_mix() {return _num_sil_mix;}
  int num_sil_states() {return _num_sil_states;}
  float sil_self_trans_prob() {return _sil_self_trans_prob;}
  float transition_alpha() {return _transition_alpha;}
  float mix_alpha() {return _mix_alpha;}
  float gaussian_a0() {return _gaussian_a0;}
  float gaussian_k0() {return _gaussian_k0;}
  float alpha_pi() {return _alpha_pi;}
  float cluster_transition_alpha() {return _cluster_transition_alpha;}
  float cluster_transition_gamma() {return _cluster_transition_gamma;}
  vector<float> gaussian_b0() {return _gaussian_b0;}
  vector<float> gaussian_u0() {return _gaussian_u0;}
  void print();
 private:
  int _num_sil_states;
  int _num_sil_mix;
  float _sil_self_trans_prob;
  int _state_num;
  int _max_duration;
  int _cluster_num;
  int _mix_num;
  int _dim;
  int _weak_limit;
  float _transition_alpha;
  float _mix_alpha;
  float _gaussian_a0;
  float _gaussian_k0;
  float _alpha_pi; // Alpha for initial probs
  float _cluster_transition_alpha; 
  float _cluster_transition_gamma; // for sampling beta. See Appendix C of [1]
  bool _parallel;
  bool _precompute;
  vector<float> _gaussian_b0;
  vector<float> _gaussian_u0;
};

#endif

// Refernces: [1] Emily Fox, Bayesian Nonparametric Learning of Complex Dynamical Phenomena
