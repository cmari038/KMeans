#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>
#include <input.h>
#include <chrono>
using namespace std;
/*
-k num_cluster: an integer specifying the number of clusters
-d dims: an integer specifying the dimension of the points
-i inputfilename: a string specifying the input filename
-m max_num_iter: an integer specifying the maximum number of iterations
-t threshold: a float specifying the threshold for convergence test.
-c: a flag to control the output of your program. If -c is specified, your program should output the centroids of all clusters. If -c is not specified, your program should output the labels of all points. See details below.
-s seed: an integer specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process. See details below.
*/

int closestCentroid(vector<float> &point, vector<vector<float>> &centroids, int dim, int clusters) {
    float min = INFINITY;
    int minIndex = 0;
    for(int j = 0; j < clusters; j++) {
        float distance = 0;
        for(int i = 0; i < dim; i++) {
            distance += pow(point.at(i) - centroids.at(j).at(i), 2);
        }
        distance = sqrtf(distance);
        if(distance < min) {
            min = distance;
            minIndex = j;
        }
    }
    //return centroids.at(minIndex);
    return minIndex;
}

void updateCentroids(map<int, vector<vector<float>>> &labels,  vector<vector<float>> &centroids, int dim, int clusters) {
    vector<float> oneD_point;
    for(int i = 0; i < clusters; i++) {
            oneD_point.assign(dim, 0.0f);
            for(int j = 0; j < labels[i].size(); j++) {
                //int length = clusters[centroids.at(i)].at(j).size();
                for(int k = 0; k < dim; k++) {
                    oneD_point[k] += labels[i].at(j).at(k) / labels[i].size();
                }
            }
            centroids[i] = oneD_point;
        }
}

bool convergence(vector<vector<float>> &centroids, vector<vector<float>> &oldCentroids, float threshold, int dim, int k) {
    int check = 0;
    for(int i = 0; i < k; i++) {
        float distance = 0;
        for(int j = 0; j < dim; j++) {
            distance += pow(oldCentroids.at(i).at(j) - centroids.at(i).at(j), 2);
        }
         distance = sqrtf(distance);
         if(distance <= threshold) {
            check++;
         }
    }

    if(check == k) {return true;}

    else{return false;}
}

int main(int argc, char* argv[]) {
    string feature;
    istringstream data;
    int numFeatures = 0;
    vector<vector<float>> features;
    ifstream inputFile(argv[3]);
    int k = stoi(argv[1]);
    int m = stoi(argv[4]);
    float t = stoi(argv[5]);
    int dim = stoi(argv[2]);
    int seed = stoi(argv[7]);
    bool c = stoi(argv[6]);
    srand(seed);


    vector<float> dataPoint;
    while(getline(inputFile, feature)) {
        stringstream data(feature);
        dataPoint.clear();
        float coord;
        numFeatures++;
        while(data >> coord) {
            dataPoint.push_back(coord);
        }
        features.push_back(dataPoint);
    }

    vector<vector<float>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }

    //input(argv[3], features, centroids, k);

    int iter = 0;
    vector<vector<float>> oldCentroids;
    map<int, vector<vector<float>>> labels;
    vector<int> centroidAssignments;
    bool done = false;

    //auto start = chrono::high_resolution_clock::now();
    chrono::milliseconds average{0};

    while(!done) {
        auto start = chrono::high_resolution_clock::now();
        oldCentroids = centroids;
        iter++;
        
        labels.clear();
        centroidAssignments.clear();
        for(int i = 0; i < numFeatures; i++) {
            int nearestCentroid = closestCentroid(features.at(i), centroids, dim, k);
            centroidAssignments.push_back(nearestCentroid);
            labels[nearestCentroid].push_back(features.at(i));
        }

        updateCentroids(labels, centroids, dim, k);
        done = iter > m || convergence(centroids, oldCentroids, t, dim, k);
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
        average += diff;
    }

    auto averageTime = average / iter;
    printf("%d,%lf\n", iter, averageTime);
    
    if(c) {
        for (int clusterId = 0; clusterId < k; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < dim; d++) {printf("%lf ", centroids.at(clusterId).at(d));}
            printf("\n");
        }
    }

    else {
        printf("clusters:");
        for (int p=0; p < numFeatures; p++) {printf(" %d", centroidAssignments.at(p));}
    }


}