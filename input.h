#ifndef _INPUT_H
#define _INPUT_H
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>
using namespace std;

void input(string fileName, vector<vector<float>> &features, vector<vector<float>> &centroids, int k) {
    ifstream inputFile(fileName);
    vector<float> dataPoint;
    int numFeatures;
    string feature;
    while(getline(inputFile, feature)) {
        stringstream data(feature);
        dataPoint.clear();
        int coord;
        numFeatures++;
        while(data >> coord) {
            dataPoint.push_back(coord);
        }
        features.push_back(dataPoint);
    }

    //vector<vector<float>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }
}
#endif