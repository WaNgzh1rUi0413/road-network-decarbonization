#include <iostream>
// #include <string>
#include <vector>
// #include <math.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <cerrno>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define INVALID -1 /* Represents an invalid value. */
#define WAS_IN_QUEUE                                                           \
  -7 /* Shows that the node was in the queue before. (7 is for luck.) */
#define INF 1.0e13
#define ZERO 1.0e-13
using namespace std;
class CNode // Network node
{
public:
  int ID;
  string Name;
  int Origin_ID = -1;
  vector<int> IncomingLink;
  vector<int> OutgoingLink;
  // CNode() : Origin_ID(-1) {}
};
class CLink // Directed road link
{
public:
  int ID;
  CNode *pInNode;
  CNode *pOutNode;
  double FreeFlowTravelTime;
  double Capacity;
  double Alpha = 0.15;
  double Power = 4.0;
  double Length;
  double MaxSpeed;
};
class CPath {
public:
  int *plinkArray;
  int n = 0;
};
class COrigin {
public:
  int ID;
  CNode *pOriginNode;
  vector<int> DestinationNode;
  vector<double> ODDemand;
  //*********************************
  vector<CPath> *PathSet;
  vector<double> *PathFlow;
};
vector<CNode> m_Node;
vector<CLink> m_Link;
vector<COrigin> m_Origin;
double *LinkFlow;
double *LinkCost;
double *LinkCostDiff;
double *LinkFreeTravelTime;
double *LinkCostCoef;
double *LinkCostDiffCoef;
//*********************************************
int *ODIncludedLinkIndex;
int ODIncludedLinkNum;
double **ODPairGap;
int **ViolationODPairIndex;
double MaxODGap;
int *m_nViolation;
int SumViolation;
//*********************************************
double *ShortestPathCost; // label list
int *ShortestPathParent;  // pre list
//*********************************************
// Configurable parameters
//*********************************************
int FIRST_THRU_NODE = 1; // CR1791;Phi1526
string DataGroup = "";
// Ensure output directories exist before writing.
static bool EnsureDir(const std::string &path) {
#ifdef _WIN32
  int rc = _mkdir(path.c_str());
#else
  int rc = mkdir(path.c_str(), 0755);
#endif
  if (rc == 0)
    return true;
  return errno == EEXIST;
}
static string GetDataDir(const string &fileName) {
  if (fileName.find("modified_") != string::npos) {
    if (!DataGroup.empty()) {
      return "revised_network/" + DataGroup + "/";
    }
    return "revised_network/";
  }
  return "network/";
}
static string GetFlowRootDir();
static string GetIntermediateDataDir(const string &fileName) {
  if (fileName.find("modified_") != string::npos) {
    return GetDataDir(fileName) + "traffic_assignment_input/";
  }
  return GetFlowRootDir() + "/traffic_assignment_input/";
}
static string GetFlowRootDir() {
  if (!DataGroup.empty()) {
    return "flow/" + DataGroup;
  }
  return "flow/original";
}
//*********************************************
int InnerIterNum = 1000;
double step = 0.3;
double MaxUEGap = 1.0e-6;
double DemandMul = 10;
// Demand multiplier applied to every OD pair.
//*****************************General*****************************
void ReadData(string fileName);
void NetworkInitialization();
void GetShortestPath(int orinode);
void UpdateLinkTravelCostAndDiff();
double GetUEGap();
double ObjectiveValue();
//*****************************GP*****************************
void GPUpdatePathSet(bool bFlowUpdate);
void GPMethodInitilization();
void GPFlowAdjustforOneODPair(COrigin *pOrigin, int des, bool bWithoutCheck);
int GPGetODFlowGap();
void UE_GPMethod();
void ReadData(string fileName) {
  CNode *pNode = NULL;
  CLink *pLink = NULL;
  COrigin *pOrigin = NULL;
  string lineStr;
  string inlineStr;
  stringstream stream;
  int intData;
  double doubleData;
  int count;
  string dataDir = GetIntermediateDataDir(fileName);

  // read fileName_node.tntp
  ifstream nodeFile(dataDir + fileName + "new_node.csv");
  if (nodeFile.is_open()) {
    count = 0;
    while (getline(nodeFile, lineStr))
    {
      pNode = new CNode();
      stringstream ss(lineStr);
      getline(ss, inlineStr, '\t');
      pNode->Name = inlineStr;
      pNode->ID = count;
      count++;
      m_Node.push_back(*pNode);
    }
    nodeFile.close();
    printf("Read node data finish!\n");
  } else {
    printf("Read node data wrong!\n");
  }

  // read fileName_link.csv
  ifstream linkFile(dataDir + fileName + "new_link.csv");
  count = 0;
  if (linkFile.is_open()) {
    while (getline(linkFile, lineStr))
    {
      pLink = new CLink();
      pLink->ID = count;
      stringstream ss(lineStr);

      getline(ss, inlineStr, ','); // Origin
      intData = stoi(inlineStr);
      pLink->pInNode = &m_Node[intData - 1];

      getline(ss, inlineStr, ','); // Destination
      intData = stoi(inlineStr);
      pLink->pOutNode = &m_Node[intData - 1];

      getline(ss, inlineStr, ','); // Length
      doubleData = stod(inlineStr);
      pLink->Length = doubleData;

      getline(ss, inlineStr, ','); // Capacity
      doubleData = stod(inlineStr);
      pLink->Capacity = doubleData;

      getline(ss, inlineStr, ','); // FFT
      doubleData = stod(inlineStr);
      pLink->FreeFlowTravelTime = doubleData * 3600;
      // pLink->FreeFlowTravelTime = pLink->Length / 16.6666666;

      getline(ss, inlineStr, ','); // Speed limit
      doubleData = stod(inlineStr);
      pLink->MaxSpeed = doubleData;

      getline(ss, inlineStr, ','); // alpha
      doubleData = stod(inlineStr);
      pLink->Alpha = doubleData;

      getline(ss, inlineStr, ','); // power
      doubleData = stoi(inlineStr);
      pLink->Power = doubleData;

      m_Link.push_back(*pLink);
      count++;
    }
    linkFile.close();
    printf("Read link data finish!\n");
  } else {
    printf("Read link data wrong!\n");
  }

  // read fileName_od.csv
  ifstream odFile(dataDir + fileName + "new_od.csv");
  if (odFile.is_open()) {

    double TDemand = 0;
    while (getline(odFile, lineStr))
    {
      stringstream ss(lineStr);
      getline(ss, inlineStr, ',');
      stream << inlineStr;
      stream >> intData;
      pNode = &m_Node[intData - 1];
      stream.clear();

      if (pNode->Origin_ID == -1) {
        pOrigin = new COrigin();
        pOrigin->ID = m_Origin.size();
        pOrigin->pOriginNode = pNode;
        pNode->Origin_ID = pOrigin->ID;
        m_Origin.push_back(*pOrigin);
        pOrigin = &m_Origin[pNode->Origin_ID];
      } else {
        pOrigin = &m_Origin[pNode->Origin_ID];
      }
      getline(ss, inlineStr, ',');
      stream << inlineStr;
      stream >> intData;
      pNode = &m_Node[intData - 1];
      stream.clear();

      getline(ss, inlineStr, ',');
      doubleData = stod(inlineStr);

      if (doubleData > 0) {
        pOrigin->DestinationNode.push_back(pNode->ID);
        pOrigin->ODDemand.push_back(doubleData * DemandMul);
      }
      TDemand += doubleData * DemandMul;
    }
    odFile.close();
    printf("Read origin data finish!\n");
    cout << setprecision(10) << "TotalDemand = " << TDemand << endl;
  } else {
    printf("Read origin data wrong!\n");
  }
}

void NetworkInitialization() {
  CLink *pLink;
  double temp;
  //********************************************
  LinkCostDiff = new double[m_Link.size()];
  LinkFlow = new double[m_Link.size()];
  LinkCost = new double[m_Link.size()];
  LinkFreeTravelTime = new double[m_Link.size()];
  LinkCostCoef = new double[m_Link.size()];
  LinkCostDiffCoef = new double[m_Link.size()];
  //********************************************
  ShortestPathCost = new double[m_Node.size()];
  ShortestPathParent = new int[m_Node.size()];
  ODIncludedLinkIndex = new int[m_Link.size()];
  for (int link = 0; link < m_Link.size(); link++) {
    pLink = &m_Link.at(link);
    pLink->pInNode->OutgoingLink.push_back(
        link);
    pLink->pOutNode->IncomingLink.push_back(link);
    LinkFreeTravelTime[link] =
        pLink->FreeFlowTravelTime;
    temp = pow(pLink->Capacity, pLink->Power);
    LinkCostCoef[link] = LinkFreeTravelTime[link] * pLink->Alpha / temp;
    LinkCostDiffCoef[link] =
        LinkFreeTravelTime[link] * pLink->Alpha * pLink->Power / temp;
  }
  //***************************************************
}

void GetShortestPath(int orinode) {
  int m_nNode = m_Node.size();
  int index, node, tempnode;
  CNode *pNode;
  CLink *pLink;
  bool *binCheckList =
      new bool[m_nNode](); // Whether each node is currently in the queue.
  bool *bscanStatus = new bool[m_nNode]();
  int *checkList = new int[m_nNode](); // Circular node queue.
  int start = 0, end = 1; // Queue head and first empty position.
  for (node = 0; node < m_nNode;
       node++) // for ( node=1; node<=no_nodes; node++)
  {
    ShortestPathCost[node] =
        INF; // Initialize shortest-path costs to infinity.
    ShortestPathParent[node] =
        -1; // Initialize shortest-path predecessor links.
            // PredLink[node] = INVALID;
    binCheckList[node] = false;
  }
  ShortestPathCost[orinode] = 0;
  checkList[start] = orinode;
  while (start != end) {
    if (start >= m_nNode)
      start = 0;
    node = checkList[start];
    start++;
    pNode = &m_Node[node];
    if (node >= FIRST_THRU_NODE - 1 || node == orinode) {
      for (index = 0; index < pNode->OutgoingLink.size(); index++) {
        pLink = &m_Link[pNode->OutgoingLink
                            [index]];
        tempnode = pLink->pOutNode->ID;
        if (ShortestPathCost[tempnode] >
            ShortestPathCost[node] + LinkCost[pLink->ID]) {
          ShortestPathCost[tempnode] =
              ShortestPathCost[node] + LinkCost[pLink->ID];
          ShortestPathParent[tempnode] = pLink->ID;
          if (!binCheckList[tempnode])
          {
            binCheckList[tempnode] = true;
            if (bscanStatus
                    [tempnode])
            {
              start--;
              if (start < 0)
                start += m_nNode;
              checkList[start] = tempnode;
            } else
            {
              if (end >= m_nNode)
                end = 0;
              checkList[end] = tempnode;
              end++;
              bscanStatus[tempnode] = true;
            }
          }
        }
      }
      binCheckList[node] = false;
    }
  }
  delete[] checkList;
  delete[] bscanStatus;
  delete[] binCheckList;
}

void UpdateLinkTravelCostAndDiff() {
  double temp, temp1;
  for (int link = 0; link < m_Link.size(); link++) {
    temp1 = LinkFlow[link];
    temp = temp1 * temp1 * temp1;
    LinkCost[link] =
        LinkFreeTravelTime[link] + LinkCostCoef[link] * temp * temp1;
    LinkCostDiff[link] = LinkCostDiffCoef[link] * temp; // pow=4
  }
}

double GetUEGap() // Compute the user-equilibrium gap.
{
  long double link_systemcost = 0.0, od_systemcost = 0.0;
  int ori, link, des;
  COrigin *pOrigin;
  for (link = 0; link < m_Link.size(); link++) {
    link_systemcost += LinkFlow[link] * LinkCost[link];
  }
  for (ori = 0; ori < m_Origin.size(); ori++) {
    pOrigin = &m_Origin.at(ori);
    GetShortestPath(pOrigin->pOriginNode->ID);
    for (des = 0; des < pOrigin->DestinationNode.size(); des++)
      od_systemcost += pOrigin->ODDemand.at(des) *
                       ShortestPathCost[pOrigin->DestinationNode.at(des)];
  }
  return 1 - od_systemcost / link_systemcost;
}

#pragma region "GP"
void GPUpdatePathSet(bool bFlowUpdate) // Update the active path set.
{
  COrigin *pOrigin;
  int *pReversePath = new int[m_Node.size()];
  int pathlinknum, templink, tempnode;
  CLink *pTempLink;
  bool bPathExist, bPathEqual;
  CPath *pPath;
  CPath TempPath;
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    pOrigin = &m_Origin.at(ori);
    GetShortestPath(pOrigin->pOriginNode->ID);
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      //****************************
      pathlinknum = 0;
      tempnode = pOrigin->DestinationNode[des];
      templink = ShortestPathParent[tempnode];
      while (templink > -1) {
        pReversePath[pathlinknum] = templink;
        pathlinknum++;
        pTempLink = &m_Link[templink];
        tempnode = pTempLink->pInNode->ID;
        templink = ShortestPathParent[tempnode];
      }
      pPath = new CPath();
      pPath->n = pathlinknum;
      pPath->plinkArray = new int[pathlinknum];
      for (int pathlink = 0; pathlink < pathlinknum; pathlink++)
        pPath->plinkArray[pathlink] =
            pReversePath[pathlinknum - pathlink -
                         1];
      //*****************************************
      // Check the existence of the route
      bPathExist = false;
      for (int path = 0; path < pOrigin->PathSet[des].size(); path++) {
        TempPath = pOrigin->PathSet[des][path];
        bPathEqual = true;
        if (TempPath.n != pathlinknum)
          bPathEqual = false;
        else {
          for (int pathlink = 0; pathlink < pathlinknum; pathlink++) {
            if (pPath->plinkArray[pathlink] != TempPath.plinkArray[pathlink]) {
              bPathEqual = false;
              break;
            }
          }
        }
        if (bPathEqual) {
          bPathExist = true;
          break;
        }
      }
      if (!bPathExist) {
        pOrigin->PathSet[des].push_back(*pPath);
        pOrigin->PathFlow[des].push_back(0.0);
        if (bFlowUpdate)
        {
          GPFlowAdjustforOneODPair(pOrigin, des, true);
        }
      }
    }
  }
  delete[] pReversePath;
}

void GPMethodInitilization() // Initialize the path-based assignment algorithm.
{

  double start = 0, finish = 0;
  COrigin *pOrigin;
  start = omp_get_wtime();
  for (int link = 0; link < m_Link.size(); link++)
    LinkCost[link] = LinkFreeTravelTime[link];
  ODPairGap = new double *[m_Origin.size()];
  ViolationODPairIndex = new int *[m_Origin.size()];
  m_nViolation = new int[m_Origin.size()];
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    pOrigin = &m_Origin.at(ori);
    ODPairGap[ori] = new double[pOrigin->DestinationNode.size()];
    ViolationODPairIndex[ori] = new int[pOrigin->DestinationNode.size()];
    pOrigin->PathSet = new vector<CPath>[pOrigin->DestinationNode.size()];
    pOrigin->PathFlow = new vector<double>[pOrigin->DestinationNode.size()];
  }
  // All-or-nothing assignment.

  GPUpdatePathSet(false);

  CPath Path;
  for (int link = 0; link < m_Link.size(); link++)
    LinkFlow[link] = 0.0;
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    pOrigin = &m_Origin.at(ori);
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      pOrigin->PathFlow[des].at(0) = pOrigin->ODDemand.at(des);
      for (int pathindex = 0; pathindex < pOrigin->PathSet[des].size();
           pathindex++)
      {
        Path = pOrigin->PathSet[des].at(pathindex);
        for (int pathlink = 0; pathlink < Path.n; pathlink++)
          LinkFlow[Path.plinkArray[pathlink]] +=
              pOrigin->PathFlow[des].at(pathindex);
      }
    }
  }
  UpdateLinkTravelCostAndDiff();
  finish = omp_get_wtime();
  double UEGap = GetUEGap();
  cout << 0 << "," << setprecision(10) << UEGap << ", " << (finish - start)
       << endl;
}

void GPFlowAdjustforOneODPair(COrigin *pOrigin, int des, bool bWithoutCheck) {
  int ori = pOrigin->ID;
  if (pOrigin->PathSet[des].size() > 1) {
    ODIncludedLinkNum = 0;
    int spathindex = 0;
    int link = 0;
    CPath Path;
    //*********************************************
    // Update path information
    bool *bLink = new bool[m_Link.size()]();
    int m_nPath = pOrigin->PathSet[des].size();
    double *ODPathCost = new double[m_nPath]();
    double *ODPathCostDiff = new double[m_nPath]();
    double *ODPathFlow = new double[m_nPath]();
    int *bRemoved = new int[m_nPath]();
    double ODTotalCost = 0.0, minpathcost = INF;

    // update path cost
    for (int path = 0; path < m_nPath; path++) {
      Path = pOrigin->PathSet[des].at(path);
      for (int pathlink = 0; pathlink < Path.n; pathlink++) {
        link = Path.plinkArray[pathlink];
        ODPathCost[path] += LinkCost[link];
        ODPathCostDiff[path] += LinkCostDiff[link];
        if (!bLink[link]) {
          ODIncludedLinkIndex[ODIncludedLinkNum] =
              link;
          bLink[link] = true;
          ODIncludedLinkNum++;
        }
      }
      if (minpathcost > ODPathCost[path]) {
        minpathcost = ODPathCost[path];
        spathindex = path;
      }
      ODTotalCost += ODPathCost[path] * pOrigin->PathFlow[des].at(path);
      if (ODPathCostDiff[path] < ZERO) //  ???????????????
        ODPathCostDiff[path] = ZERO;
    }
    //**************************************************
    // Find the least-cost path in the current path set.

    ODPairGap[ori][des] =
        1 - minpathcost * pOrigin->ODDemand.at(des) / ODTotalCost;
    // GetObjVal();
    double spTempFlow = pOrigin->ODDemand.at(des);
    double tempPF = 0;
    if (ODPairGap[ori][des] > MaxODGap ||
        bWithoutCheck)
    {
      // update path flow
      for (int path = 0; path < m_nPath; path++) {
        if (path != spathindex) {
          tempPF = pOrigin->PathFlow[des].at(path) -
                   step * (ODPathCost[path] - ODPathCost[spathindex]) /
                       ODPathCostDiff[path];
          if (tempPF <= ZERO) {
            bRemoved[path] = 1;
            ODPathFlow[path] = 0.0;
          } else {
            spTempFlow -= tempPF;
            ODPathFlow[path] = tempPF;
          }
        }
      }

      ODPathFlow[spathindex] = spTempFlow;
      //***************************************************
      // update link flow, cost, and cost diffential
      for (int path = m_nPath - 1; path >= 0; path--) {
        Path = pOrigin->PathSet[des].at(path);
        for (int pathlink = 0; pathlink < Path.n; pathlink++) {
          LinkFlow[Path.plinkArray[pathlink]] +=
              (ODPathFlow[path] -
               pOrigin->PathFlow[des].at(path)); // GaussSeidel
          // if (abs(LinkFlow[Path.plinkArray[pathlink]]) < 1e-8)
          //   LinkFlow[Path.plinkArray[pathlink]] = 0.0;
        }
        if (bRemoved[path]) {
          delete[] pOrigin->PathSet[des].at(path).plinkArray;
          pOrigin->PathFlow[des].erase(pOrigin->PathFlow[des].begin() + path);
          pOrigin->PathSet[des].erase(pOrigin->PathSet[des].begin() + path);
        } else
          pOrigin->PathFlow[des][path] = ODPathFlow[path];
      }
      double temp = 0, temp1 = 0;
      for (int linkindex = 0; linkindex < ODIncludedLinkNum; linkindex++) {
        link = ODIncludedLinkIndex[linkindex];
        temp1 = LinkFlow[link];
        temp = temp1 * temp1 * temp1;
        LinkCost[link] =
            LinkFreeTravelTime[link] + LinkCostCoef[link] * temp * temp1;
        LinkCostDiff[link] = LinkCostDiffCoef[link] * temp;
      }
      SumViolation++;
    }
    delete[] bLink;
    delete[] ODPathCost;
    delete[] ODPathCostDiff;
    delete[] ODPathFlow;
    delete[] bRemoved;
  } else
    ODPairGap[ori][des] = 0.0;
}

int GPGetODFlowGap() { //
  // update path cost
  int totalviolation = 0;
  COrigin *pOrigin = NULL;
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    pOrigin = &m_Origin.at(ori);
    m_nViolation[ori] = 0;
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      if (pOrigin->PathSet[des].size() > 1) {
        CPath Path;
        int link = 0;
        int spathindex = 0;
        ODIncludedLinkNum = 0;
        bool *bLink = new bool[m_Link.size()]();
        //*********************************************
        // Update path information
        int m_nPath = pOrigin->PathSet[des].size();
        double *ODPathCost = new double[m_nPath]();
        double *ODPathCostDiff = new double[m_nPath]();
        double *ODPathFlow = new double[m_nPath]();
        int *bRemoved = new int[m_nPath]();
        double ODTotalCost = 0.0, minpathcost = INF;

        // update path cost
        for (int path = 0; path < m_nPath; path++) {
          Path = pOrigin->PathSet[des][path];
          for (int pathlink = 0; pathlink < Path.n; pathlink++) {
            link = Path.plinkArray[pathlink];
            ODPathCost[path] += LinkCost[link];
            ODPathCostDiff[path] += LinkCostDiff[link];
            if (!bLink[link]) {
              ODIncludedLinkIndex[ODIncludedLinkNum] =
                  link;
              bLink[link] = true;
              ODIncludedLinkNum++;
            }
          }
          if (minpathcost > ODPathCost[path]) {
            minpathcost = ODPathCost[path];
            spathindex = path;
          }
          ODTotalCost += ODPathCost[path] * pOrigin->PathFlow[des].at(path);
          if (ODPathCostDiff[path] < ZERO)
            ODPathCostDiff[path] = ZERO;
        }
        ODPairGap[ori][des] =
            1 - minpathcost * pOrigin->ODDemand.at(des) / ODTotalCost;
        double spTempFlow = pOrigin->ODDemand.at(des);
        double tempPF = 0;
        // ************************************************
        if (ODPairGap[ori][des] > MaxODGap) {
          ViolationODPairIndex[ori][m_nViolation[ori]] = des;
          m_nViolation[ori]++;
          // update path flow
          for (int path = 0; path < m_nPath; path++) {
            if (path != spathindex) {
              tempPF = pOrigin->PathFlow[des].at(path) -
                       step * (ODPathCost[path] - ODPathCost[spathindex]) /
                           ODPathCostDiff[path];
              if (tempPF <= 0) {
                bRemoved[path] = 1;
                ODPathFlow[path] = 0.0;
              } else {
                spTempFlow -= tempPF;
                ODPathFlow[path] = tempPF;
              }
            }
          }
          ODPathFlow[spathindex] = spTempFlow;
          //***************************************************
          for (int path = m_nPath - 1; path >= 0; path--) {
            Path = pOrigin->PathSet[des].at(path);
            for (int pathlink = 0; pathlink < Path.n; pathlink++) {
              LinkFlow[Path.plinkArray[pathlink]] +=
                  ODPathFlow[path] - pOrigin->PathFlow[des].at(path);
              // if (abs(LinkFlow[Path.plinkArray[pathlink]]) < 1e-8)
              //   LinkFlow[Path.plinkArray[pathlink]] = 0.0;
            }
            if (bRemoved[path]) {
              delete[] pOrigin->PathSet[des].at(path).plinkArray;
              pOrigin->PathFlow[des].erase(pOrigin->PathFlow[des].begin() +
                                           path);
              pOrigin->PathSet[des].erase(pOrigin->PathSet[des].begin() + path);
            } else
              pOrigin->PathFlow[des][path] = ODPathFlow[path];
          }
          double temp, temp1;
          for (int linkindex = 0; linkindex < ODIncludedLinkNum; linkindex++) {
            link = ODIncludedLinkIndex[linkindex];
            temp1 = LinkFlow[link];
            temp = temp1 * temp1 * temp1;
            LinkCost[link] =
                LinkFreeTravelTime[link] + LinkCostCoef[link] * temp * temp1;
            LinkCostDiff[link] = LinkCostDiffCoef[link] * temp;
          }
        }
        delete[] bLink;
        delete[] ODPathCost;
        delete[] ODPathCostDiff;
        delete[] ODPathFlow;
        delete[] bRemoved;
      } else
        ODPairGap[ori][des] = 0.0;
    }
    totalviolation += m_nViolation[ori];
  }
  return totalviolation;
}

double R(string fileName) {
  int MaxPathNumber = 0;
  double fenzi, fenmu;
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    COrigin *pOrigin = &m_Origin[ori];
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      int m_nPath = pOrigin->PathSet[des].size();
      if (m_nPath > MaxPathNumber)
        MaxPathNumber = m_nPath;
    }
  }
  int *PathNumber;
  PathNumber = new int[MaxPathNumber];
  for (int i = 0; i < MaxPathNumber; i++) {
    PathNumber[i] = 0;
  }
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    COrigin *pOrigin = &m_Origin[ori];
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      int m_nPath = pOrigin->PathSet[des].size();
      PathNumber[m_nPath - 1]++;
    }
  }
  ofstream ofile;
  ofile.open("path.txt", ios::app);
  ofile << fileName << endl;
  for (int i = 0; i < MaxPathNumber; i++) {
    ofile << i + 1 << "," << PathNumber[i] << endl;
    fenzi += PathNumber[i] * (i + 1) * (i + 1);
    fenmu += PathNumber[i] * (i + 1);
  }
  ofile.close();
  return fenzi / fenmu;
}

double averagepath() {
  double pathnum = 0;
  int ODnum = 0;
  for (int ori = 0; ori < m_Origin.size(); ori++) {
    COrigin *pOrigin = &m_Origin[ori];
    for (int des = 0; des < pOrigin->DestinationNode.size(); des++) {
      int m_nPath = pOrigin->PathSet[des].size();
      for (int path = 0; path < m_nPath; path++) {
        if (pOrigin->PathFlow[des].at(path) > 0.1 * pOrigin->ODDemand.at(des)) {
          pathnum++;
        }
      }
      ODnum++;
    }
  }
  return pathnum / ODnum;
}

void UE_GPMethod(string fileName) // Solve user equilibrium with the gradient-projection method.
{
  COrigin *pOrigin;
  int nOutLoop = 0;
  double obj = 0;
  int des;
  MaxODGap = 1.0e-3;
  int GapCheckIterNum = 100;
  double GapRate = 2.0;
  double UEGap = 1.0;
  double start, finish, CPUTime;
  NetworkInitialization();
  start = omp_get_wtime();
  GPMethodInitilization(); // 9s

  double iftime = 0, elsetime = 0, PG_period = 0, RG_period = 0, PFA_period = 0,
         PFA_beg;
  while (UEGap > MaxUEGap && nOutLoop < 1000) //&& CPUTime < 4800
  {
    nOutLoop++;
    double PG_beg = omp_get_wtime();
    GPUpdatePathSet(true);
    double PG_end = omp_get_wtime();
    PG_period += PG_end - PG_beg;
    int a = 0, b = 0, iter = 0;
    PFA_beg = omp_get_wtime();
    for (iter = 0; iter < InnerIterNum; iter++) {
      SumViolation = 0;
      if (iter % GapCheckIterNum == 0) {
        double if_beg = omp_get_wtime();
        int num = GPGetODFlowGap();
        double if_end = omp_get_wtime();
        iftime += (if_end - if_beg);
        if (num == 0)
          break;
      }
      double else_beg = omp_get_wtime();
      for (int ori = 0; ori < m_Origin.size(); ori++) {
        pOrigin = &m_Origin.at(ori);
        for (int desindex = 0; desindex < m_nViolation[ori]; desindex++) {
          des = ViolationODPairIndex[ori][desindex];
          if (ODPairGap[ori][des] > MaxODGap) {
            GPFlowAdjustforOneODPair(pOrigin, des, false);
          }
        }
      }
      double else_end = omp_get_wtime();
      elsetime += (else_end - else_beg);
      if (SumViolation == 0)
        break;
    }
    double PFA_end = omp_get_wtime();
    PFA_period += PFA_end - PFA_beg;
    UEGap = GetUEGap();
    MaxODGap = UEGap / GapRate;
    double finish = omp_get_wtime();
    RG_period += finish - PFA_end;
    CPUTime = (finish - start);
    obj = ObjectiveValue();
    // cout << setprecision(20) << obj << ", " << nOutLoop << "," <<
    // setprecision(10) << UEGap << ", " << CPUTime << ", " << endl;
  }
  // double r = R(fileName);
  double apn = averagepath();

  double high_con_link_num = 0;
  for (int i = 0; i < m_Link.size(); i++) {
    if (LinkFlow[i] > 0.75 * m_Link[i].Capacity)
      high_con_link_num = high_con_link_num + 1;
  }
  double high_con_link_per = high_con_link_num / m_Link.size();

  printf("%s\t", "wtime:");
  printf("%f\n", CPUTime);
  printf("%s\t", "nOutLoop:");
  printf("%d\n", nOutLoop);
  printf("%s\t", "UEGap:");
  printf("%e\n", UEGap);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  ofstream ofile;
  string flowDir;
  string flowSuffix;
  size_t suffixPos = fileName.find("modified_");
  if (suffixPos != string::npos) {
    flowSuffix = fileName.substr(suffixPos);
  }
  if (DemandMul == 1.0)
    flowDir = GetFlowRootDir() + "/flow1";
  if (DemandMul == 2.0)
    flowDir = GetFlowRootDir() + "/flow2";
  if (DemandMul == 3.0)
    flowDir = GetFlowRootDir() + "/flow3";
  if (DemandMul == 4.0)
    flowDir = GetFlowRootDir() + "/flow4";
  if (DemandMul == 5.0)
    flowDir = GetFlowRootDir() + "/flow5";
  if (DemandMul == 6.0)
    flowDir = GetFlowRootDir() + "/flow6";
  if (DemandMul == 7.0)
    flowDir = GetFlowRootDir() + "/flow7";
  if (DemandMul == 8.0)
    flowDir = GetFlowRootDir() + "/flow8";
  if (DemandMul == 9.0)
    flowDir = GetFlowRootDir() + "/flow9";
  if (DemandMul == 10.0)
    flowDir = GetFlowRootDir() + "/flow10";

  if (!flowDir.empty() && !flowSuffix.empty()) {
    flowDir = flowDir + "_" + flowSuffix;
  }

  if (!flowDir.empty()) {
    EnsureDir("flow");
    EnsureDir(GetFlowRootDir());
    EnsureDir(flowDir);
    string filename1 = flowDir + "/" + fileName + "_flow.csv";
    ofile.open(filename1);
    if (!ofile.is_open()) {
      cerr << "Cannot open flow file: " << filename1 << endl;
    } else {
      ofile << "AvePathNum=" << apn << endl;
      ofile << "ConLinkPer=" << high_con_link_per << endl;
      for (int i = 0; i < m_Link.size(); i++) {
        double speed = 0;
        speed = m_Link[i].Length / LinkCost[i];
        ofile << m_Link[i].pInNode->ID + 1 << "," << m_Link[i].pOutNode->ID + 1
              << "," << m_Link[i].Length << "," << m_Link[i].Capacity << ","
              << m_Link[i].FreeFlowTravelTime << "," << m_Link[i].MaxSpeed << ","
              << m_Link[i].Alpha << "," << m_Link[i].Power
              << ","
              //<< fixed << setprecision(5)
              << LinkFlow[i] << "," << LinkCost[i] << "," << speed << endl;
      }
      ofile.close();
    }
  }

  if (fileName.find("modified_") != string::npos) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    ofstream ofile2;
    string dataDir = GetDataDir(fileName);
    string dataDirNoSlash = dataDir;
    if (!dataDirNoSlash.empty() && dataDirNoSlash.back() == '/')
      dataDirNoSlash.pop_back();
    EnsureDir(dataDirNoSlash);
    string filename2 = dataDir + fileName + "_flow.csv";
    ofile2.open(filename2);
    ofile2 << "AvePathNum=" << apn << endl;
    ofile2 << "ConLinkPer=" << high_con_link_per << endl;
    for (int i = 0; i < m_Link.size(); i++) {
      double speed = 0;
      speed = m_Link[i].Length / LinkCost[i];
      ofile2 << m_Link[i].pInNode->ID + 1 << "," << m_Link[i].pOutNode->ID + 1
             << "," << m_Link[i].Length << "," << m_Link[i].Capacity << ","
             << m_Link[i].FreeFlowTravelTime << "," << m_Link[i].MaxSpeed << ","
             << m_Link[i].Alpha << "," << m_Link[i].Power
             << ","
             << LinkFlow[i] << "," << LinkCost[i] << "," << speed << endl;
    }
    ofile2.close();
  }
}
double ObjectiveValue() {
  double obj = 0.0;
  double link_integral = 0.0;
  CLink *pLink;
  for (int link = 0; link < m_Link.size(); link++) {
    pLink = &m_Link.at(link);
    link_integral = LinkFreeTravelTime[link] * LinkFlow[link] +
                    LinkCostCoef[link] * pow(LinkFlow[link], 5) / 5;
    obj += link_integral;
  }
  return obj;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_string> <demand_multiplier> [data_group]" << std::endl;
    return 1;
  }

  string fileName(argv[1]); // Phi Birm CR CS
  cout << "////////// " << fileName << " //////////" << endl;
  string dd(argv[2]);
  if (argc >= 4) {
    DataGroup = argv[3];
  }
  DemandMul = stoi(dd);
  ReadData(fileName);
  UE_GPMethod(fileName);
}
