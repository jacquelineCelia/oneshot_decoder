/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: datum.h 
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#ifndef DATUM_H
#define DATUM_H

#include <vector>

#include "config.h"
#include "segment.h"
#include "bound.h"

using namespace std;

// A datum represents one training data point, which 
// includes the hidden letter sequence and the observed
// speech data. 
// A datum only keeps the relationship of words
// and letters, segments and bounds.
// It does not do any actions.
// This should enable parallel computation.

class Datum {
 public:
  Datum(Config*);
  // Access Functions:
  vector<Segment*>& segments() {return _segments;}
  vector<Bound*>& bounds() {return _bounds;}
  string tag() {return _tag;}
  void set_bounds(vector<Bound*>& bounds) {_bounds = bounds;}
  void set_segments(vector<Segment*>& segments) {_segments = segments;}
  void set_tag(const string& tag) {_tag = tag;}
  void ClearSegs(); 
  void Save(const string&);
  void Save();
  void set_corrupted(bool t) {_corrupted = t;}
  void set_score(float s) {_score = s;}
  bool is_corrupted() {return _corrupted;}
  int FindNextBoundary(int);
  int FindBoundSearchBoundary(int);
  float score() {return _score;}
  ~Datum();
 private:
  vector<Segment*> _segments;
  vector<Bound*> _bounds;
  string _tag;
  Config* _config;
  bool _corrupted;
  float _score;
};

#endif
