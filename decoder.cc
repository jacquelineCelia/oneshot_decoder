/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: decoder.cc 
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <queue>
#include <cstring>

#include "bound.h"
#include "decoder.h"
#include "toolkit.h"

#define DEBUG false 

using namespace std;

Decoder::Decoder(Config* config) {
    _config = config;
}

void Decoder::ComputeSegProbGivenCluster(Datum* datum, \
        vector<Cluster*> clusters, \
        float*** segment_prob_given_cluster) {
    for (int i = 0; i < (int) clusters.size(); ++i) {
        segment_prob_given_cluster[i] = \
            clusters[i] -> ConstructSegProbTable(datum -> bounds());
        /*
        for (size_t b1 = 0; b1 < (datum -> bounds()).size(); ++b1) {
            for (size_t b2 = 0; b2 < (datum -> bounds()).size(); ++b2) {
                cout << "segment_prob_given_cluster[" << i << "][" << b1 << "][" << b2 << "]: " << segment_prob_given_cluster[i][b1][b2] << " ";
            }
            cout << endl;
        }
        cout << endl;
        */
    }
}

void Decoder::MessageBackward(Datum* datum, \
        float* pi, float** A, \
        float*** segment_prob_given_cluster, \
            ProbList<int>** B, ProbList<int>** Bstar) {
    /*
    for (int i = 0; i < _config -> cluster_num(); ++i) {
        cout << "pi[" << i << "]: " << pi[i] << endl;
    }
    for (int i = 0; i < _config -> cluster_num(); ++i) {
        for (int j = 0; j < _config -> cluster_num(); ++j) {
            cout << "A[" << i << "][" << j << "]: " << A[i][j] << " ";
        }
        cout << endl;
    }
    */
    vector<Bound*> bounds = datum -> bounds();
    int b = bounds.size();
    int c = _config -> weak_limit();
    // Initialization
    for (int i = 0; i <= c; ++i) {
        B[i][b].push_back(0, -1);
    }
    int max_duration = _config -> max_duration();
    int total_frame_num = 0;
    vector<int> accumulated_frame_num(b, 0);
    for (int i = 0; i < b; ++i) {
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_num[i] = total_frame_num;
    }
    // Compute B and Bstar
    for (int i = b - 1; i >= 0; --i) {
        for (int j = c; j > 0; --j) {
            // Compute Bstar
            int start_frame = i == 0 ? 0 : accumulated_frame_num[i - 1];
            for (int k = i + 1; k <= b && ((accumulated_frame_num[k - 1] - start_frame) <= max_duration || k == i + 1); ++k) { 
                Bstar[j][i].push_back(B[j][k].value() + segment_prob_given_cluster[j - 1][i][k - 1], k);
            }
        }
        for (int j = c; j >= 0; --j) {
            // Compute B
            if ((j > 0 && i > 0) || (j == 0 && i == 0)){
                if (j == 0 && i == 0) {
                    for (int k = 1; k <= c; ++k) {
                        B[j][i].push_back(pi[k - 1] + Bstar[k][i].value(), k);
                    }
                }
                else {
                    for (int k = 1; k <= c; ++k) {
                        B[j][i].push_back(A[j - 1][k - 1] + Bstar[k][i].value(), k);
                    }
                }
            }
        }
    }
}

void Decoder::DecodeForward(Datum* datum, \
        ProbList<int>** B, ProbList<int>** Bstar) {
    vector<Bound*> bounds = datum -> bounds();
    int b = bounds.size();
    int i = 0, j = 0;
    while (j < b) {
        /*
        vector<float> probs = B[i][j].probs();
        for (size_t z = 0; z < probs.size(); ++z) {
            cout << "probs[" << z << "]: " << probs[z] << " ";
        }
        cout << endl;
        */
        int next_i = B[i][j].index(FindIndexForMaxEntry(B[i][j].probs()));
        int next_j = Bstar[next_i][j].index(FindIndexForMaxEntry(Bstar[next_i][j].probs()));
        // Create corresponding segment
        int id = next_i - 1;
        Segment* segment = new Segment(_config, id);
        (datum -> segments()).push_back(segment);
        for (int p = j; p < next_j; ++p) {
            segment -> push_back(bounds[p]);
        }
        i = next_i;
        j = next_j;
    }
}

int Decoder::FindIndexForMaxEntry(vector<float> a) {
    float max = a[0];
    int index = 0; 
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] > max) {
            max = a[i];
            index = i;
        }
    }
    return index;
}

float Decoder::ComputeLikelihoodGivenFixedDecoding(Datum* datum, vector<GMM*> gmm, float** A) {
    vector<Bound*> bounds = datum -> bounds();
    int b = bounds.size();
    int total_frame_num = 0;
    vector<int> accumulated_frame_nums(b, 0);
    int start_frame = bounds[0] -> start_frame();
    int end_frame = bounds[b - 1] -> end_frame();
    vector<float*> frames(end_frame - start_frame + 1, NULL); 
    for (unsigned int i = 0 ; i < b; ++i) {
        if (!(_config -> precompute())) {
            vector<float*> bound_data = bounds[i] -> data();
            copy(bound_data.begin(), \
                bound_data.end(), frames.begin() + total_frame_num);
        }
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_nums[i] = total_frame_num;
    }
    float** frame_prob_for_each_state;
    frame_prob_for_each_state = new float* [gmm.size()];
    for (size_t i = 0 ; i < gmm.size(); ++i) {
        frame_prob_for_each_state[i] = new float[total_frame_num];
        if (!(_config -> precompute())) {
            gmm[i] -> ComputeLikehood(frames, frame_prob_for_each_state[i]);
        }
        else {
            gmm[i] -> ComputeLikehood(start_frame, end_frame, frame_prob_for_each_state[i]);
        }
    }
    ToolKit _toolkit;
    vector<float> cur_prob(gmm.size(), MIN_PROB_VALUE);
    for (int i = 0; i < total_frame_num; ++i) {
        if (!i) {
            cur_prob[0] = frame_prob_for_each_state[0][i];
        }
        else {
            vector<float> next_prob(gmm.size(), MIN_PROB_VALUE);
            for (size_t k = 0; k < gmm.size(); ++k) {
                vector<float> summands;
                for (size_t l = 0; l <= k; ++l) {
                    summands.push_back(cur_prob[l] + A[l][k]);
                }
                next_prob[k] = _toolkit.SumLogs(summands) + frame_prob_for_each_state[k][i];
            } 
            cur_prob = next_prob;
        }
    }    
    vector<float> next_prob(gmm.size(), MIN_PROB_VALUE);
    for (size_t i = 0; i < gmm.size(); ++i) {
        next_prob[i] = cur_prob[i] + A[i][gmm.size()];
    }
    return _toolkit.SumLogs(next_prob);
}

void Decoder::GetMostLikelySegments(Datum* datum, Model* model) {
    vector<Cluster*> clusters = model -> clusters();
    int b = (datum -> bounds()).size();
    int c = _config -> weak_limit();
    // Compute P(b|c) 
    if (DEBUG) {
        cout << "Computing P(b|c)" << endl;
    }
    float*** segment_prob_given_cluster = new float** [c];
    ComputeSegProbGivenCluster(datum, clusters, segment_prob_given_cluster);
    // Compute B and Bstar
    if (DEBUG) {
        cout << "Allocating space for B and Bstar" << endl;
    }
    ProbList<int> **Bstar, **B; 
    B = new ProbList<int>* [c + 1];
    Bstar = new ProbList<int>* [c + 1];
    for (int i = 0 ; i <= c; ++i) {
        B[i] = new ProbList<int> [b + 1];
        Bstar[i] = new ProbList<int> [b + 1];
    }
    if (DEBUG) {
        cout << "Message Backward" << endl;
    }
    MessageBackward(datum, model -> pi(), model -> A(), \
            segment_prob_given_cluster, B, Bstar);
    // Sample forward
    if (DEBUG) {
        cout << "Sample Forward" << endl;
    }
    DecodeForward(datum, B, Bstar);
    // Delete allocated memory: B, Bstar, ProbList, segment_prob_given_cluster
    if (DEBUG) {
        cout << "deleting B and Bstar" << endl;
    }
    for (int i = 0; i <= c; ++i) {
        delete[] B[i];
        delete[] Bstar[i];
    }
    delete[] B;
    delete[] Bstar;
    if (DEBUG) {
        cout << "deleting segment_prob_given_cluster" << endl;
    }
    for (int i = 0; i < _config -> weak_limit(); ++i) {
        for (int j = 0; j < b; ++j) {
            delete[] segment_prob_given_cluster[i][j];
        }
        delete[] segment_prob_given_cluster[i]; 
    }
    delete[] segment_prob_given_cluster;
} 
