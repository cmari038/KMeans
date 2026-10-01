#include <string>
using namespace std;
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>
/*
-k num_cluster: an integer specifying the number of clusters
-d dims: an integer specifying the dimension of the points
-i inputfilename: a string specifying the input filename
-m max_num_iter: an integer specifying the maximum number of iterations
-t threshold: a double specifying the threshold for convergence test.
-c: a flag to control the output of your program. If -c is specified, your program should output the centroids of all clusters. If -c is not specified, your program should output the labels of all points. See details below.
-s seed: an integer specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process. See details below.
*/

int closestCentroid(vector<double> &point,  vector<vector<double>> &centroids, int dim, int clusters) {
    double min = INFINITY;
    int minIndex = 0;
    for(int j = 0; j < clusters; j++) {
        double distance = 0;
        for(int i = 0; i < dim; i++) {
            distance += pow(point.at(i) - centroids.at(j).at(i), 2);
        }
        distance = sqrt(distance);
        if(distance < min) {
            min = distance;
            minIndex = j;
        }
    }
    //return centroids.at(minIndex);
    return minIndex;
}

void updateCentroids(map<int, vector<vector<double>>> &labels,  vector<vector<double>> &centroids, int dim, int clusters) {
    vector<double> oneD_point;
    for(int i = 0; i < clusters; i++) {
            oneD_point.resize(labels[i].size(),0);
            for(int j = 0; j < labels[i].size(); j++) {
                //int length = clusters[centroids.at(i)].at(j).size();
                for(int k = 0; k < dim; k++) {
                    oneD_point[k] += labels[i].at(j).at(k) / dim;
                }
            }
            centroids.pop_back();
            centroids.push_back(oneD_point);
        }
}

bool convergence(vector<vector<double>> &centroids, vector<vector<double>> &oldCentroids, int threshold, int dim, int k) {
    double distance = 0;
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < dim; j++) {
            distance += pow(oldCentroids.at(i).at(j) - centroids.at(i).at(j), 2);
        }
    }
        distance = sqrt(distance);

    if(distance <= threshold) {return true;}

    else{return false;};
}

int main(int argc, char* argv[]) {
    string feature;
    istringstream data;
    int numFeatures = 0;
    vector<vector<double>> features;
    ifstream inputFile(argv[3]);
    int k = stoi(argv[1]);
    int m = stoi(argv[4]);
    int t = stoi(argv[5]);
    int dim = stoi(argv[2]);
    int seed = stoi(argv[6]);
    srand(seed);


    vector<double> dataPoint;
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

    vector<vector<double>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }

    int iter = 0;
    vector<vector<double>> oldCentroids;
    map<int, vector<vector<double>>> labels;
    //vector<int> labels;
    bool done = false;

    while(!done) {
        oldCentroids = centroids;
        iter++;
        
        labels.clear();
        for(int i = 0; i < features.size(); i++) {
            int nearestCentroid = closestCentroid(features.at(i), centroids, dim, k);
            labels[nearestCentroid].push_back(features.at(i));
        }

        updateCentroids(labels, centroids, dim, k);
        done = iter > m || convergence(centroids, oldCentroids, t, dim, k);
    }
    
}