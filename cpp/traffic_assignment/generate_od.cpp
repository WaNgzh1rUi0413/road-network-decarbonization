#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <map>
#include <random>
#include <string>
#include <cerrno>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace std;

static bool EnsureDir(const string& path) {
#ifdef _WIN32
  return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <input_string>" << std::endl;
    return 1;
  }
  string fileName(argv[1]);
  string dataGroup = argc >= 3 ? argv[2] : "";
  string lineStr;
  string inlineStr;
  stringstream stream;
  double node[10];
  string odDir;
  if (fileName.find("modified_") != string::npos) {
    odDir = "revised_network/";
    if (!dataGroup.empty()) {
      odDir += dataGroup + "/";
    }
    odDir += "traffic_assignment_input/";
  } else {
    string flowGroup = dataGroup.empty() ? "original" : dataGroup;
    EnsureDir("flow");
    EnsureDir("flow/" + flowGroup);
    odDir = "flow/" + flowGroup + "/traffic_assignment_input/";
  }
  if (!EnsureDir(odDir)) {
    cerr << "Cannot create traffic assignment input directory: " << odDir << endl;
    return 1;
  }
  string centerBase = fileName;
  const string originalSuffix = "-original";
  if (centerBase.size() > originalSuffix.size() &&
      centerBase.compare(centerBase.size() - originalSuffix.size(), originalSuffix.size(), originalSuffix) == 0) {
    centerBase = centerBase.substr(0, centerBase.size() - originalSuffix.size());
  }

  string centerResultDir = "k_center/results/";
  if (!dataGroup.empty()) {
    centerResultDir += dataGroup + "/";
  }
  string centerPathRevised = centerResultDir + centerBase + "_kcenter_k10_binary_search_revised.csv";
  string centerPathPlain = "k_center/results/original/" + centerBase + "_kcenter_k10_binary_search.csv";

  ifstream nodeFile(centerPathRevised);
  if (!nodeFile.is_open()) {
    nodeFile.open(centerPathPlain);
  }

  if (nodeFile.is_open()) {
    int count = 0;
    getline(nodeFile, lineStr);
    while (getline(nodeFile, lineStr)) {
      stringstream ss(lineStr);
      getline(ss, inlineStr, ',');
      node[count] = stoi(inlineStr) + 1;
      count++;
    }
    nodeFile.close();
    printf("Read node data finish!\n");
  } else {
    cerr << "Read node data wrong: " << centerPathRevised << " or " << centerPathPlain << endl;
    return 1;
  }

  ofstream ofile;
  ofile.open(odDir + fileName + "new_od.csv");
  if (!ofile.is_open()) {
    cerr << "Cannot open OD output file: " << odDir + fileName + "new_od.csv" << endl;
    return 1;
  }
  double ODDemand = 500;
  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
        if(i!=j)
          ofile << node[i] << "," << node[j] << ","  << ODDemand << endl;
    }
  }
  ofile.close();
  cout << "Generate OD finish" << endl;
}
