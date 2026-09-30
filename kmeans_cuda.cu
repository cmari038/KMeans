#include <string>
using namespace std;
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>

__global__ void closestCentroid(double* centroids, double* point, double* labels, int dim, int clusters) {
    double min = INFINITY;
    int minCluster = 0;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x < clusters * dim) {
        for(int i = 0; i < clusters; i++) {
            double distance = 0;
            for(int j = 0; j < dim; j++) {
                distance += sqrt((centroids[i * dim + j] - point[x * dim + j]) * (centroids[i * dim + j] - point[x * dim + j]));
            }
            if(distance < min) {
                    min = distance;
                    minCluster = j;
            }
        }
    }
    labels[x] = minCluster;
}

__global__ void updateCentroids(double* centroids, double* point, double* labels, double* sums, int* cluster_counts, int dim, int clusters) {
    
    int thread = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < labels * dim) {
        int cluster = labels[idx];
        atomicAdd(&cluster_counts[cluster], 1);
        for(int i = 0; i < dim; i++) {
            atomicAdd(&sums[cluster * k + i], &point[idx * k + i]);
        }
        for(int i = 0; i < dim; i++) {
            centroids[idx * dim + i] = sums[idx * dim + i] / cluster_counts[idx];
        }
    }
}

bool convergence(double* centroids, double* oldCentroids, int threshold, int dim) {
    double distance = 0;
    for(int i = 0; i < centroids.size(); i++) {
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
    int numFeatures;
    vector<double> features;
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
        while(data >> coord) {dataPoint.push_back(coord);}
        features.push_back(dataPoint);
    }

    vector<vector<double>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }

    double* cuda_features;
    double* cuda_centroids;
    double* sums;
    int* cluster_counts;
    double* host_features = (double*)malloc(numFeatures*dim*sizeof(double));
    double* host_centroids = (double*)malloc(k*dim*sizeof(double));
    double* host_sums = (double*)malloc(k*sizeof(double));
    int* host_cluster_sizes = (double*)malloc(k*sizeof(int));

    //flatten
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < dim; j++) {
            cuda_centroids[i*j] = centroids.at(i).at(j);
        }
    }

    for(int i = 0; i < features.size(); i++) {
        for(int j = 0; j < dim; j++) {
            cuda_features[i*j] = (features.at(i).at(j));
        }
    }

    cudaMalloc(&cuda_centroids, k*dim*sizeof(double));
    cudaMalloc(&cuda_features, numFeatures*dim*sizeof(double));
    cudaMalloc(&cluster_counts, k*sizeof(int));
    cudaMalloc(&sums, k*sizeof(double));
    cudaMemcpy(sums, host_sums, k*sizeof(double), cudaMemcpyHosttoDevice);
    cudaMemcpy(cuda_features, host_points,  numFeatures*dim*sizeof(double), cudaMemcpyHosttoDevice);
    cudaMemcpy(cuda_centroids, host_centroids,  k*dim*sizeof(double), cudaMemcpyHosttoDevice);
    cudaMemcpy(cluster_counts, host_cluster_sizes,  k*sizeof(int), cudaMemcpyHosttoDevice);

    int* labels[numFeatures];
    int* host_labels = (int*)malloc(numFeatures*sizeof(int));
    cudaMalloc(&labels, numFeatures*sizeof(int));
    cudaMemcpy(labels, host_labels,  numFeatures*sizeof(int), cudaMemcpyHosttoDevice);

    int iter = 0;
    bool done = false;

    while(!done) {
        oldCentroids = cuda_centroids;
        iter++;

        closestCentroid<<<15, 64>>>(cuda_centroids, cuda_features, labels, dim, k);

        updateCentroids<<<15, 64>>>(cuda_centroids, cuda_features, labels, sums, cluster_counts, dim, k);
        done = iter > m || convergence(centroids, oldCentroids, t, dim);
    }

    cudaFree(cuda_features);
    cudaFree(cuda_centroids);
    cudaFree(labels);

    free(host_features);
    free(host_centroids);
    free(host_labels);
}