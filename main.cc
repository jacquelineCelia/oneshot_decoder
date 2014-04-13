/* -*- C++ -*-
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved
 *	FILE: main.cc   				                                *
 *  Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				    *
 *  April 2014							                            *
*********************************************************************/

#include <iostream>
#include <string>
#include <cstdlib>

#include "config.h"
#include "manager.h"

using namespace std;

void usage();

int main(int argc, char* argv[]) {
    int mode = atoi(argv[2]);
    if (mode == 3) {
        if (argc != 17) {
            usage();
            return -1;
        }
        else {
            string fn_example_index = argv[4];
            string fn_example_data = argv[6];
            string fn_query_list = argv[8];
            string fn_out = argv[10];
            string fn_config = argv[12];
            string basedir = argv[14];
            string snapshot = argv[16];
            Config config;
            if (!config.Load(fn_config)) {
                cout << "Cannot load configuration file." 
                    << " Check " << fn_config << endl;
            }
            else {
                cout << "Configuration file loaded successfully." << endl;
            }
            Manager manager(&config);
            if (!manager.LoadData(fn_example_index, fn_example_data, fn_out)) {
                cout << "Cannot load bounds" 
                    << " Check " << fn_example_index << " " << fn_example_data << " " << fn_out << endl; 
            }
            else {
                cout << "Example loaded successfully." << endl;
            }
            if (!manager.LoadData(fn_query_list)) {
                cout << "Cannot load queries."
                    << "Check " << fn_query_list << endl;
            }
            else {
                cout << "Queries loaded successfully!." << endl; 
            }
            manager.InitializeModel(snapshot);
            manager.ModifyModelPi();
            cout << "Decoding..." << endl;
            manager.Decode();
            cout << "Ranking..." << endl;
            manager.Rank();
            cout << "Saving alignments ... " << endl;
            manager.SaveData();
        }
    }
    else if (mode == 2) {
        if (argc != 13) {
            usage();
            return -1;
        } 
        string fn_index = argv[4];
        string fn_data = argv[6];
        string fn_out = argv[8];
        string fn_config = argv[10];
        string fn_snapshot = argv[12];
        Config config;
        if (!config.Load(fn_config)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        Manager manager(&config);
        if (!manager.LoadData(fn_index, fn_data, fn_out)) {
            cout << "Cannot load bounds" 
                << " Check " << fn_index << " " << fn_data << " " << fn_out << endl; 
        }
        else {
            cout << "Data loaded successfully." << endl;
        }
        manager.InitializeModel(fn_snapshot);
        cout << "Decoding..." << endl;
        manager.Decode();
        manager.SaveData();
    }
    else {
        if (argc != 17 && argc != 19 && argc != 15) {
            usage();
            return -1;
        }
        string fn_list = argv[4];
        string fn_config = argv[6];
        int n_iter = atoi(argv[8]);
        string fn_gaussian = argv[10];
        string basedir = argv[12];
        int batch_size = atoi(argv[14]);
        string fn_sil = ""; 
        string fn_snapshot = "";

        if (argc == 17) {
            fn_sil = argv[16];
        }
        if (argc == 19) {
            fn_snapshot = argv[18];
        }

        Config config;
        if (!config.Load(fn_config, fn_gaussian)) {
            cout << "Cannot load configuration file." 
                << " Check " << fn_config << endl;
        }
        else {
            cout << "Configuration file loaded successfully." << endl;
        }
        config.print();
        Manager manager(&config);
        if (mode == 0) {
            if (fn_snapshot == "") {
                cout << "No file model is specified. Need a previous snapshot file" << endl;
            }
            else {
                manager.InitializeModel(fn_snapshot);
                cout << "Model has shown successfully" << endl;
            }
        }
        else if (mode == 1) {
            if (!manager.LoadData(fn_list)) {
            cout << "Cannot load bounds" 
                << " Check " << fn_list << endl; 
            }
            else {
                cout << "Data loaded successfully." << endl;
            }
            if (fn_sil != "") {
                if (!manager.LoadSilenceModel(fn_sil)) {
                    cout << "Cannot load silence model " 
                        << "Check " << fn_sil << endl;
                }
            }
            else {
                cout << "Training model without using silence model" << endl;
            }
            if (fn_snapshot == "") {
                manager.InitializeModel();
            }
            else {
                manager.InitializeModel(fn_snapshot);
            }
            manager.Inference(batch_size, n_iter, basedir);
        }
        else {
            cout << "Undefined mode: [0: read model; 1: training]" << endl;
        }
        cout << "Congrats, it's done!" << endl;
    }
    return 0;
}

void usage() {
    cout << "gibbs -m [0: read model; 1: training; 2: decoding] -l data_list -c configuration -n num_iteration " 
        << "-g gaussian_prior -b basedir -z batch_size -s silence_model -snapshot snapshot_file" << endl;
    cout << "gibbs -m 2 -index fn_index -data fn_data -out fn_out -c configuration -b basedir -snapshot snapshot_file" << endl;
    cout << "gibbs -m 3 -example_index fn_index -example_data fn_data -query_list query_list  -out fn_out -c config -b basedir -snapshot snapshot_file" << endl; 
}
