#pragma once
#include <list>
#include <vector>

using namespace std;

extern bool clustering(list<vector<float>>& regFacialFeature1, list<vector<float>>& regFacialFeature2, list<int>& clusterID1, list<int>& clusterID2, list<int>& selectedImage1, list<int>& selectedImage2, float Scth);