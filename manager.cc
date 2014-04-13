/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: manager.cc   				                                *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <set>
#include <omp.h>
#include <ctime>

#include "cluster.h"
#include "gmm.h"
#include "manager.h"
#include "counter.h"
#include "decoder.h"

Manager::Manager(Config* config) {
    _config = config;
    _model = new Model(_config);
    _total_frame_num = 0;
}

bool Manager::LoadSilenceModel(string& fn_sil) {
    ifstream fsil(fn_sil.c_str(), ios::binary);
    if (!fsil.is_open()) {
        return false;
    }
    // load silence model
    int num_state = _config -> num_sil_states();
    int num_mixture = _config -> num_sil_mix();
    Cluster* sil_cluster = new Cluster(_config, num_state, num_mixture, 0); 
    vector<vector<float> > trans_probs;
    for (int i = 0; i < num_state; ++i) {
        vector<float> trans_prob(num_state + 1, 0);
        trans_prob[0]= log(_config -> sil_self_trans_prob());
        trans_prob[1] = log(1 - _config -> sil_self_trans_prob());
        trans_probs.push_back(trans_prob);
    }
    sil_cluster -> set_transition_probs(trans_probs); 
    for (int i = 0 ; i < num_state; ++i) {
        vector<float> weights;
        for (int m = 0; m < num_mixture; ++m) {
            float weight;
            fsil.read(reinterpret_cast<char*> (&weight), sizeof(float));
            weights.push_back(weight);
            vector<float> mean(_config -> dim(), 0);
            vector<float> pre(_config -> dim(), 0);
            fsil.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * _config -> dim());
            fsil.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * _config -> dim());
            (sil_cluster -> emission(i)).mixture(m).set_mean(mean);
            (sil_cluster -> emission(i)).mixture(m).set_pre(pre);
            (sil_cluster -> emission(i)).mixture(m).set_det();
        }
        (sil_cluster -> emission(i)).set_weight(weights);
    }
    fsil.close();
    sil_cluster -> set_is_fixed(true);
    _model -> AddSilenceCluster(sil_cluster);
    return true;
}

void Manager::InitializeModel() {
    _model -> Initialize();
    // cout << "There are " << (_model -> clusters()).size() << " clusters." << endl;
}

void Manager::InitializeModel(const string& fn_snapshot) {
    _model -> LoadSnapshot(fn_snapshot);
    // cout << "There are " << (_model -> clusters()).size() << " clusters." << endl;
}

void Manager::ParallelInference(int batch_size, int n_iter, const string& basedir) {
    /*
    cout << "Doing parallel inference" << endl;
    Counter global_counter(_config);
    Sampler global_sampler(_config);
    // Initialization
    global_sampler.SampleModelParams(_model);

    for (int i = 0; i <= n_iter; ++i) {
        time_t start_time = time(NULL);        
        vector<Datum*> batch = i < 10 ? _datum : randomizer.GetBatch(batch_size);
        LoadDataIntoMatrix(batch);
        if (_config -> precompute()) {
            _model -> PreCompute(&_features[0], _total_frame_num);
        }
        omp_set_num_threads(64);
    #pragma omp parallel 
    { 
        Sampler local_sampler(_config);
        Counter local_counter(_config);
    #pragma omp for schedule(dynamic, 1)
        for (vector<Datum*>::iterator d_iter = batch.begin(); \
                d_iter < batch.end(); ++d_iter) {
            if (!((*d_iter) -> is_corrupted())) {
                local_sampler.SampleSegments((*d_iter), _model, &local_counter); 
            }
            else {
                cout << "!!!!!!!!!! " << (*d_iter) -> tag() << " is corrupted." << endl;  
            }
        }
    #pragma omp critical
        global_counter += local_counter;
    }
        global_sampler.SampleModelParams(_model, &global_counter);
        if (i % 1000 == 0) {
            stringstream n;
            n << i;
            string output_dir = basedir + "/" + n.str() + "/";
            SaveData(output_dir);
            SaveModel(output_dir);
        }
        cout << "Done the " << i << "th iteration." << endl;
        time_t end_time = time(NULL);
        cout << "It took " << end_time - start_time << " seconds to finish one iteration" << endl;
    }
    */
}

void Manager::SerielInference(int batch_size, int n_iter, const string& basedir) {
    /*
    Counter global_counter(_config);
    Sampler global_sampler(_config);
    // Initialization
    cout << "Sampling global model" << endl;
    global_sampler.SampleModelParams(_model);

    for (int i = 0; i <= n_iter; ++i) {
        time_t start_time = time(NULL); 
        vector<Datum*> batch = i < 10 ? _datum : randomizer.GetBatch(batch_size);
        LoadDataIntoMatrix(batch);
        if (_config -> precompute()) {
            cout << "precomputing" << endl;
            _model -> PreCompute(&_features[0], _total_frame_num);
        }
        for (vector<Datum*>::iterator d_iter = batch.begin(); \
                d_iter != batch.end(); ++d_iter) {
            if (!((*d_iter) -> is_corrupted())) {
                if (i == 0) {
                    cout << "Inferencing " << (*d_iter) -> tag() << endl;
                }
                global_sampler.SampleSegments((*d_iter), _model, &global_counter); 
            }
            else {
                cout << "!!!!!!!!!! " << (*d_iter) -> tag() << " is corrupted." << endl;  
            }
        }
        cout << "Sampling model" << endl;
        global_sampler.SampleModelParams(_model, &global_counter);
        if (i % 1000 == 0) {
            stringstream n;
            n << i;
            string output_dir = basedir + "/" + n.str() + "/";
            SaveData(output_dir);
            SaveModel(output_dir);
        }
        cout << "Done the " << i << "th iteration." << endl;
        time_t end_time = time(NULL); 
        cout << "It took " << end_time - start_time << " seconds to finish one iteration" << endl;
    }
    */
}

void Manager::SaveModel(const string& output_dir) {
    string path = output_dir + "snapshot";
    _model -> Save(path);
}

void Manager::SaveData(const string& output_dir) {
    /*
    #pragma omp parallel
    {
        #pragma omp for schedule (dynamic, 1)
        for (int d = 0; d < (int) _datum.size(); ++d) {
            _datum[d] -> Save(output_dir);
        }
    }
    */
}

void Manager::SaveData() {
    for (int d = 0; d < (int) _datum.size(); ++d) {
        _datum[d] -> Save();
    }
}

void Manager::Inference(int batch_size, int n_iter, const string& basedir) {
    if (_config -> parallel()) {
        ParallelInference(batch_size, n_iter, basedir);
    }
    else {
        SerielInference(batch_size, n_iter, basedir);
    }
}

string Manager::GetTag(const string& s) {
   size_t found_last_slash, found_last_period;
   found_last_slash = s.find_last_of("/");
   found_last_period = s.find_last_of(".");
   return s.substr(found_last_slash + 1, \
     found_last_period - 1 - found_last_slash);
}

bool Manager::LoadData(string& fn_data_list) {
    ifstream flist(fn_data_list.c_str());
    while (flist.good()) {
        string fn_index;
        string fn_data;
        string fn_out;
        flist >> fn_index;
        flist >> fn_data;
        flist >> fn_out; 
        if (fn_index != "" && fn_data != "" && fn_out != "") {
            string tag = fn_out; 
            ifstream findex(fn_index.c_str());
            ifstream fdata(fn_data.c_str(), ios::binary);
            if (!findex.is_open()) {
                cout << "Cannot open " << fn_index << endl;
                return false;
            }
            if (!fdata.is_open()) {
                cout << "Cannot open " << fn_data << endl;
                return false;
            }
            // cout << "loading " << fn_data << endl;
            Datum* datum = new Datum(_config);
            datum -> set_tag(tag);
            LoadBounds(datum, findex, fdata);
            findex.close();
            fdata.close();
            _queries.push_back(datum);
        }
    }
    flist.close();
    return true;
}

void Manager::LoadBounds(Datum* datum, ifstream& findex, ifstream& fdata) {
     vector<Bound*> bounds;
     int total_frame;
     findex >> total_frame;
     int start_frame = 0;
     int end_frame = 0;
     int label = -2; // >= 0: is labeled, is boundary. == -1: is boundary. == -2 : normal
     while (end_frame != total_frame - 1) {
         findex >> start_frame;
         findex >> end_frame;
         findex >> label;
         if (start_frame > end_frame) {
            datum -> set_corrupted(true);
            break;
         }
         int frame_num = end_frame - start_frame + 1;
         Bound* bound = new Bound(_config);
         // float** data is deleted in bound.cc
         float** data = new float* [frame_num];
         for (int i = 0; i < frame_num; ++i) {
             data[i] = new float [_config -> dim()];
             fdata.read(reinterpret_cast<char*> (data[i]), \
                     sizeof(float) * _config -> dim());
         }
         bound -> set_data(data, frame_num);
         if (label >= 0) {
             if (bounds.size() > 0) {
                bounds[bounds.size() - 1] -> set_is_boundary(true);
             }
             bound -> set_is_labeled(true);
             bound -> set_is_boundary(true);
             bound -> set_label(label);
         }
         if (label == -1) {
             bound -> set_is_boundary(true);
         }
         bound -> set_start_frame(start_frame + _total_frame_num);
         bound -> set_end_frame(end_frame + _total_frame_num);
         bounds.push_back(bound);
     }
     if (datum -> is_corrupted()) {
         vector<Bound*>::iterator b_iter = bounds.begin();
         for (; b_iter != bounds.end(); ++b_iter) {
            delete *b_iter;
         }
     }
     else {
        _total_frame_num += total_frame;
        bounds[bounds.size() - 1] -> set_is_boundary(true);
        datum -> set_bounds(bounds);
     }
}

void Manager::LoadDataIntoMatrix(vector<Datum*>& batch) {
    _features.clear();
    _total_frame_num = 0;
    vector<Datum*>::iterator d_iter = batch.begin();
    for (; d_iter != batch.end(); ++d_iter) {
        vector<Bound*> bounds = (*d_iter) -> bounds(); 
        vector<Bound*>::iterator b_iter = bounds.begin();
        for (; b_iter != bounds.end(); ++b_iter) {
            vector<float*> f = (*b_iter) -> data(); 
            for (unsigned int i = 0; i < f.size(); ++i) {
                _features.push_back(f[i]);
            }
            (*b_iter) -> set_start_frame(_total_frame_num);
            (*b_iter) -> set_end_frame(_total_frame_num + (*b_iter) -> frame_num() - 1); 
            _total_frame_num += (*b_iter) -> frame_num(); 
        }
    }
}

bool Manager::LoadData(const string& fn_index, \
                       const string& fn_data, \
                       const string& fn_out) {
    if (fn_index != "" && fn_data != "" && fn_out != "") {
        string tag = fn_out; 
        ifstream findex(fn_index.c_str());
        ifstream fdata(fn_data.c_str(), ios::binary);
        if (!findex.is_open()) {
            cout << "Cannot open " << fn_index << endl;
            return false;
        }
        if (!fdata.is_open()) {
            cout << "Cannot open " << fn_data << endl;
            return false;
        }
        cout << "loading " << fn_data << endl;
        Datum* datum = new Datum(_config);
        datum -> set_tag(tag);
        LoadBounds(datum, findex, fdata);
        findex.close();
        fdata.close();
        _datum.push_back(datum);
    }
    return true;
}

void Manager::ModifyModelPi() {
    _model -> ModifyPi();
}

void Manager::Decode() {
    vector<Datum*> batch = _datum;
    // vector<Datum*> batch = _queries;
    LoadDataIntoMatrix(batch);
    if (_config -> precompute()) {
        _model -> PreCompute(&_features[0], _total_frame_num);
    }
    Decoder local_decoder(_config);
    for (vector<Datum*>::iterator d_iter = batch.begin(); \
            d_iter < batch.end(); ++d_iter) {
        local_decoder.GetMostLikelySegments((*d_iter), _model); 
    }
}

void Manager::Rank() {
    Decoder local_decoder(_config);
    for (vector<Datum*>::iterator d_iter = _queries.begin(); \
            d_iter < _queries.end(); ++d_iter) {
        local_decoder.GetMostLikelySegments((*d_iter), _model); 
    }
    vector<Cluster*> clusters = _model -> clusters();
    for (vector<Datum*>::iterator d_iter = _queries.begin(); \
            d_iter < _queries.end(); ++d_iter) {
        vector<Segment*> segments = (*d_iter) -> segments();
        vector<GMM*> gmm;
        for (size_t i = 0; i < segments.size(); ++i) {
            int id = segments[i] -> id();
            int state_num = clusters[id] -> state_num(); 
            for (int j = 0; j < state_num; ++j) {
                gmm.push_back(clusters[id] -> GMM_ptr(j));
            }
        }
        float** A = new float* [gmm.size()];
        for (size_t i = 0; i < gmm.size(); ++i) {
            A[i] = new float [gmm.size() + 1];
            for (int j = 0; j < gmm.size() + 1; ++j) {
                A[i][j] = MIN_PROB_VALUE;
            }
        }
        int offset = 0;
        for (size_t i = 0; i < segments.size(); ++i) {
            int id = segments[i] -> id();
            int state_num = clusters[id] -> state_num();
            for (int j = 0; j < state_num; ++j) {
                for (int k = j; k < state_num + 1; ++k) {
                    A[j + offset][k + offset] = clusters[id] -> transition_prob(j, k);
                }
            }
            offset += state_num;
        }
        float score = local_decoder.ComputeLikelihoodGivenFixedDecoding(_datum[0], gmm, A);
        (*d_iter) -> set_score(score);
        for (size_t i = 0; i < gmm.size(); ++i) {
            delete [] A[i];
        }
        delete [] A;
    }
    SaveRankData();
}

void Manager::SaveRankData() {
    string filename = _datum[0] -> tag();
    ofstream fout(filename.c_str());
    fout << _datum[0] -> tag() << endl;
    vector<Segment*> s = _datum[0] -> segments();
    for (size_t i = 0; i < s.size(); ++i) {
        fout << s[i] -> id() << " ";
    }
    fout << endl;
    vector<Datum*>::iterator q_iter = _queries.begin();
    for (; q_iter != _queries.end(); ++q_iter) {
        fout << (*q_iter) -> tag() << " " << (*q_iter) -> score() << endl;
        /*
        fout << (*q_iter) -> tag() << ": ";
        vector<Segment*> segs = (*q_iter) -> segments();
        for (size_t i = 0; i < segs.size(); ++i) {
            fout << segs[i] -> id() << " ";
        }
        fout << endl;
        */
    }
    fout.close();
}

Manager::~Manager() {
    vector<Datum*>::iterator d_iter = _datum.begin();
    for (; d_iter != _datum.end(); ++d_iter) {
        delete *d_iter;
    }
    _datum.clear();
    delete _model;
}
