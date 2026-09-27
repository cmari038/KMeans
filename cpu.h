#include <vector>
#include <map>
using namespace std;

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

bool convergence(vector<vector<double>> &centroids, vector<vector<double>> &oldCentroids, int threshold, int dim) {
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

void kmeans(int numFeatures, vector<vector<double>> features, int k, int m, int dim, int t) {
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
        done = iter > m || convergence(centroids, oldCentroids, t, dim);
    }
    
    
}
