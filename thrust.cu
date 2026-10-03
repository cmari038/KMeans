#include <thrust>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>
#include <chrono>
using namespace thrust;

int closestCentroid(device_vector<float> &point,  device_vector<float> &centroids, int dim, int clusters) {
    float min = INFINITY;
    int minIndex = 0;
    device_vector<float> sums(dim);
    device_vector<float> centroid(dim);
    float distance = 0;
    for(int i = 0; i < clusters; i++) {
        thrust::copy(centroids.begin() + (i*dim), centroids.begin() + (i * dim) + dim, centroid);
        thrust::transform(point.begin(), point.end(), centroid.begin(), sums.begin(), [] __device__(float x, float y) {return (x-y)*(x-y)});
        distance = thrust::reduce(sums.begin(), sums.end(), 0.0f, thrust::plus<float>());
        distance = sqrtf(distance);
        if(distance < min) {
            min = distance;
            minIndex = i;
        }
    }

    return minIndex;
}

void updateCentroids(map<int, vector<device_vector<float>>> &labels,  host_vector<float> &centroids,  int dim, int clusters) {
    device_vector<float> sums(dim);
    for(int i = 0; i < clusters; i++) {
            for(int j = 0; j < labels[i].size(); j++) {
                if(j==0) {
                    sums = labels[i].at(j);
                }
                thrust::transform(labels[i].at(j).begin(), labels[i].at(j).end(), sums.begin(), sums.begin(), thrust::plus<float>());
            }
            thrust::transform(sums.begin(), sums.end(), sums.begin(), [] __device__ (float x) {return x / labels[i].size()});
            thrust::copy(sums.begin(), sums.end(), centroids.begin() + (i*dim));
        }
}

bool convergence(host_vector<float> &centroids, host_vector<float> &oldCentroids, int threshold, int dim, int clusters) {
    device_vector<float> deviceCentroids = centroids;
    device_vector<float> deviceOldCentroids = oldCentroids;
    device_vector<float> sums(dim);
    float distance = 0;
    int check = 0;

    for(int i = 0; i < clusters; i++) {
        //distance += pow(oldCentroids.at(i).at(j) - centroids.at(i).at(j), 2);
        thrust::transform(deviceCentroids.begin() + (i * dim), deviceCentroids.begin() + (i * dim) + dim, deviceOldCentroids.begin() +  (i * dim), sums.begin(), [] __device__(float x, float y) {(x-y)*(x-y)})
        distance = sqrtf(thrust::reduce(sums.begin(), sums.end()));
        if(distance <= threshold) {check++;}
    }
    if(check == clusters) {
        return true;
    }

    else{return false;}
}

int main(int argc, char* argv[]) {
    string feature;
    istringstream data;
    int numFeatures = 0;
    std::vector<vector<float>> features;
    ifstream inputFile(argv[3]);
    int k = stoi(argv[1]);
    int m = stoi(argv[4]);
    int t = stoi(argv[5]);
    int dim = stoi(argv[2]);
    int seed = stoi(argv[7]);
    int c = stoi(argv[6]);
    srand(seed);


    std::vector<float> dataPoint;
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

    std::vector<vector<float>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }

    host_vector<float> host_centroids(k*dim);
    host_vector<float> host_features(numFeatures*dim);

    for(int i = 0; i < k; i++) {
        for(int j = 0; j < dim; j++) {
            host_centroids.at(i * dim + j) = centroids.at(i).at(j);
        }
    }

    for(int i = 0; i < features.size(); i++) {
        for(int j = 0; j < dim; j++) {
            host_features.at(i * dim + j) = (features.at(i).at(j));
        }
    }

    int iter = 0;
    host_vector<float> oldCentroids(k*dim);
    //host_vector<int> labels;
    map<int, vector<device_vector<float>>> labels;
    bool done = false;
    device_vector<float> deviceFeatures;
    device_vector<float> deviceCentroids;
    vector<int> centroidAssignments;
    chrono::milliseconds average{0};

    while(!done) {
        auto start = chrono::high_resolution_clock::now();
        oldCentroids = host_centroids;
        iter++;
        
        labels.clear();
        deviceCentroids = host_centroids;
        for(int i = 0; i < numFeatures; i++) {
            deviceFeatures.clear();
            deviceFeatures.resize(dim);
            thrust::copy(host_features.begin() + (i * dim), host_features.begin() + (i * dim) + dim, deviceFeatures.begin());
            //deviceCentroids = host_centroids;
            int nearestCentroid = closestCentroid(deviceFeatures, deviceCentroids, dim, k);
            //copy(deviceFeatures.begin(), deviceFeatures.end(), labels.begin() + i);
            labels[nearestCentroid].push_back(deviceFeatures);
            centroidAssignments.push_back(nearestCentroid);
            
            auto end = chrono::high_resolution_clock::now();
            auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
            average += diff;
        }

        updateCentroids(labels, host_centroids, dim, k);
        done = iter > m || convergence(host_centroids, oldCentroids, t, dim, k);
    }

    if(c) {
        for (int clusterId = 0; clusterId < k; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < dim; d++) {printf("%lf ", host_centroids.at(clusterId + d * k));}
            printf("\n");
        }
    }

    else {
        printf("clusters:");
        for (int p=0; p < numFeatures; p++) {printf(" %d", centroidAssignments.at(p));}
    }

    
}