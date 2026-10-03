#include <string>
using namespace std;
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>
#include <map>
#include <cmath>

__global__ void closestCentroid(float* centroids, float* point, int* labels, int dim, int clusters, int numFeatures) {
    float min = INFINITY;
    int minCluster = 0;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float shared_centroids[];
    //__shared__ float shared_point[idx * dim + dim];
    for(int i = threadIdx.x; i < clusters * dim; i+= blockDim.x) {
        shared_centroids[i] = centroids[i];
    }
    //shared_point[x] = point[x];
    __syncthreads();

    if(x < numFeatures) {
        for(int i = 0; i < clusters; i++) {
            float distance = 0;
            for(int j = 0; j < dim; j++) {
                distance += (shared_centroids[i * dim + j] - point[x * dim + j]) * (shared_centroids[i * dim + j] - point[x * dim + j]);
            }
            distance = sqrtf(distance);
            if(distance < min) {
                    min = distance;
                    minCluster = i;
            }
        }
        labels[x] = minCluster;
    }
}

__global__ void addCentroids(float* centroids, float* point, int* labels, float* sums, int* cluster_counts, int dim, int clusters, int numFeatures) {
    
    int thread = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //extern __shared__ float shared_centroids[];
    extern __shared__ char sharedMem;
    float* shared_sums = (float*)sharedMem;
    int* shared_cluster_counts = (int*)shared_sums + clusters*dim; 

    for(int i = thread; i < clusters * dim; i+= blockDim.x) {
        //shared_centroids[i] = centroids[i];
        shared_sums[i] = 0;
    }
    for(int i = thread; i < clusters; i += blockDim.x) {
        shared_cluster_counts[i] = 0;
    }
    __syncthreads();

    if(idx < numFeatures) {
        int cluster = labels[idx];
        atomicAdd(&shared_cluster_counts[cluster], 1);
        for(int i = 0; i < dim; i++) {
            atomicAdd(&shared_sums[cluster * dim + i], point[idx * dim + i]);
        }
    }

     __syncthreads();

    for(int i = thread; i < clusters * dim; i+= blockDim.x) {
        atomicAdd(&sums[i], shared_sums[i]);
    }
    for(int i = thread; i < clusters; i += blockDim.x) {
        atomicAdd(&cluster_counts[i], shared_cluster_counts[i]);
    }

}

__global__ void updateCentroids(float* centroids, float* sums, int* cluster_counts, int dim, int clusters, int numFeatures) {
    
    //int thread = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < cluster * dim) {
        //for(int i = 0; i < dim; i++) {
        centroids[idx] = sums[idx] / cluster_counts[idx / dim;];
        //}
    }
}

bool convergence(float* centroids, float* oldCentroids, int threshold, int dim, int k) {
    float distance = 0;
    int check = 0;
    for(int i = 0; i < k * dim; i++) {
        distance += pow(oldCentroids[i] - centroids[i], 2);
    }

    distance = sqrtf(distance);
    if(distance <= threshold) {return true;}

    else {return false;}
}

int main(int argc, char* argv[]) {
    string feature;
    istringstream data;
    int numFeatures = 0;
    vector<vector<float>> features;
    ifstream inputFile(argv[3]);
    int k = stoi(argv[1]);
    int m = stoi(argv[4]);
    int t = stoi(argv[5]);
    int dim = stoi(argv[2]);
    int seed = stoi(argv[7]);
    int c = stoi(argv[6]);
    srand(seed);


    vector<float> dataPoint;
    while(getline(inputFile, feature)) {
        stringstream data(feature);
        dataPoint.clear();
        float coord;
        numFeatures++;
        while(data >> coord) {dataPoint.push_back(coord);}
        features.push_back(dataPoint);
    }

    vector<vector<float>> centroids;
    for(int i = 0; i < k; i++) {
        centroids.push_back(features.at(rand() % numFeatures));
    }

    float* cuda_features;
    float* cuda_centroids;
    float* sums;
    int* cluster_counts;
    float* host_features = (float*)malloc(numFeatures*dim*sizeof(float));
    float* host_centroids = (float*)malloc(k*dim*sizeof(float));
    //float* host_sums = (float*)malloc(k*dim*sizeof(float));
    //int* host_cluster_sizes = (float*)malloc(k*sizeof(int));

    //flatten
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < dim; j++) {
            host_centroids[i * dim + j] = centroids.at(i).at(j);
        }
    }

    for(int i = 0; i < features.size(); i++) {
        for(int j = 0; j < dim; j++) {
            host_features[i * dim + j] = (features.at(i).at(j));
        }
    }

    cudaMalloc(&cuda_centroids, k*dim*sizeof(float));
    cudaMalloc(&cuda_features, numFeatures*dim*sizeof(float));
    cudaMalloc(&cluster_counts, k*sizeof(int));
    cudaMalloc(&sums, k * dim * sizeof(float));

    cudaMemcpy(cuda_features, host_features,  numFeatures*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_centroids, host_centroids,  k*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sums, 0, k * dim * sizeof(float));
    cudaMemset(cluster_counts, 0,  k * sizeof(int));

    int* labels;
    //int* host_labels = (int*)malloc(numFeatures*sizeof(int));
    cudaMalloc(&labels, numFeatures*sizeof(int));
    cudaMemset(labels, 0,  numFeatures*sizeof(int));

    int iter = 0;
    bool done = false;
    float* oldCentroids = (float*)malloc(k*dim*sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float averageTime = 0;

    while(!done) {
        cudaEventRecord(start);
        cudaMemcpy(oldCentroids, cuda_centroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost);
        iter++;

        closestCentroid<<< (numFeatures + 255)/ 256, 256 >>>(cuda_centroids, cuda_features, labels, dim, k, numFeatures);

        cudaMemset(cluster_counts, 0,  k * sizeof(int));
        cudaMemset(sums, 0,  k * dim * sizeof(float));

        addCentroids<<< (numFeatures + 255)/256, 256 >>>(cuda_centroids, cuda_features, labels, sums, cluster_counts, dim, k, numFeatures);
        updateCentroids<<< (k * dim + 255)/256, 256 >>>(cuda_centroids, sums, cluster_counts, dim, k, numFeatures);
        cudaMemcpy(host_centroids, cuda_centroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost);
        done = iter > m || convergence(host_centroids, oldCentroids, t, dim, k);

        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        averageTime += milliseconds;
    }

    averageTime = averageTime / iter;
    printf("%d,%lf\n", iter, averageTime);
    
    if(c) {
        for (int clusterId = 0; clusterId < k; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < dim; d++) {printf("%lf ", host_centroids[clusterId + dim * k]);}
            printf("\n");
        }
    }

    else {
        int* host_labels = (int*)malloc(numFeatures*sizeof(int));
        cudaMemcpy(host_labels, labels, numFeatures * sizeof(int), cudaMemcpyDeviceToHost);
        printf("clusters:");
        for (int p=0; p < numFeatures; p++) {printf(" %d", host_labels[p]);}
    }


    cudaFree(cuda_features);
    cudaFree(cuda_centroids);
    cudaFree(labels);
    cudaFree(cluster_counts);
    cudaFree(sums);

    free(host_features);
    free(host_centroids);
    //free(host_labels);
    //free(host_sums);
    //free(host_cluster_sizes);
}