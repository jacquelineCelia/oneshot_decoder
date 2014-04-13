/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: manager.h   				                                *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef MANAGER_H
#define MANAGER_H

#include <string>
#include <iostream>
#include <fstream>
#include "config.h"
#include "datum.h"
#include "model.h"
// #include "randomizer.h"
#include "bound.h"

class Manager {
 public:
  Manager(Config*);
  bool LoadData(string& fn_data_list);
  void LoadDataIntoMatrix(vector<Datum*>&);
  bool LoadData(const string&, const string&, const string&);
  void LoadBounds(Datum*, ifstream&, ifstream&);
  bool LoadSilenceModel(string&);
  void Inference(int, int, const string&);
  void ParallelInference(int, int, const string&);
  void SerielInference(int, int, const string&);
  void InitializeModel();
  void InitializeModel(const string&);
  void SaveData(const string&);
  void SaveModel(const string&);
  void Decode();
  void SaveData();
  void Rank();
  void SaveRankData();
  void ModifyModelPi();
  string GetTag(const string&);
  ~Manager();
 private:
  vector<Datum*> _datum;
  Model* _model;
  Config* _config;
  int _total_frame_num;
  vector<float*> _features;
  vector<Datum*> _queries;
  // Randomizer randomizer;
};

#endif
