#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cerrno>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace std;

static bool FileExists(const string& path) {
    ifstream f(path.c_str());
    return f.good();
}

static string GetRevisedDir(const string& dataGroup) {
    if (dataGroup.empty()) {
        return "../../revised_network/";
    }
    return "../../revised_network/" + dataGroup + "/";
}

static string ResolveInputPath(const string& fileName, const string& suffix, const string& dataGroup) {
    string revised = GetRevisedDir(dataGroup) + fileName + suffix;
    if (FileExists(revised)) {
        return revised;
    }
    return "../../network/" + fileName + suffix;
}

static bool EnsureDir(const string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

static string ResolveOutputDir(const string& fileName, const string& dataGroup) {
    if (fileName.find("modified_") != string::npos) {
        string outputDir = GetRevisedDir(dataGroup) + "traffic_assignment_input/";
        if (!EnsureDir(outputDir)) {
            cerr << "Cannot create traffic assignment input directory: " << outputDir << endl;
            return "";
        }
        return outputDir;
    }

    string flowGroup = dataGroup.empty() ? "original" : dataGroup;
    string flowRoot = "../../flow/";
    string groupDir = flowRoot + flowGroup + "/";
    EnsureDir(flowRoot);
    EnsureDir(groupDir);
    string outputDir = groupDir + "traffic_assignment_input/";
    if (!EnsureDir(outputDir)) {
        cerr << "Cannot create traffic assignment input directory: " << outputDir << endl;
        return "";
    }
    return outputDir;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <fileName>" << endl;
        return 1;
    }

    string fileName(argv[1]);
    string dataGroup = argc >= 3 ? argv[2] : "";
    string outputDir = ResolveOutputDir(fileName, dataGroup);
    if (outputDir.empty()) {
        return 1;
    }
    string lineStr;

    // Convert node IDs from zero-based to one-based indexing.
    {
        string inPath  = ResolveInputPath(fileName, "_node.csv", dataGroup);
        string outPath = outputDir + fileName + "new_node.csv";

        ifstream nodeIn(inPath.c_str());
        if (!nodeIn.is_open()) {
            cerr << "Cannot open node file: " << inPath << endl;
            return 1;
        }

        ofstream nodeOut(outPath.c_str(), ios::trunc);
        if (!nodeOut.is_open()) {
            cerr << "Cannot open output node file: " << outPath << endl;
            return 1;
        }

        while (getline(nodeIn, lineStr)) {
            if (lineStr.empty()) continue;

            stringstream ss(lineStr);
            string idStr, xStr, yStr;

            if (!getline(ss, idStr, ',')) continue;
            if (!getline(ss, xStr, ','))  continue;
            if (!getline(ss, yStr, ','))  continue;

            // Parse the source ID and shift it by one.
            double idDouble = 0.0;
            try {
                idDouble = stod(idStr);
            } catch (...) {
                // Skip malformed node rows.
                continue;
            }
            int idInt = static_cast<int>(idDouble);
            idInt += 1;

            nodeOut << idInt << "," << xStr << "," << yStr << "\n";
        }

        nodeIn.close();
        nodeOut.close();
        cout << "Finish writing new_node.csv" << endl;
    }

    // Shift link endpoints and write a bidirectional representation.
    {
        string inPath  = ResolveInputPath(fileName, "_link.csv", dataGroup);
        string outPath = outputDir + fileName + "new_link.csv";

        ifstream linkIn(inPath.c_str());
        if (!linkIn.is_open()) {
            cerr << "Cannot open link file: " << inPath << endl;
            return 1;
        }

        ofstream linkOut(outPath.c_str(), ios::trunc);
        if (!linkOut.is_open()) {
            cerr << "Cannot open output link file: " << outPath << endl;
            return 1;
        }

        while (getline(linkIn, lineStr)) {
            if (lineStr.empty()) continue;

            stringstream ss(lineStr);
            string oStr, dStr, restStr;

            // Read the origin and destination columns.
            if (!getline(ss, oStr, ',')) continue;
            if (!getline(ss, dStr, ',')) continue;

            // Preserve all remaining comma-separated attributes.
            if (!getline(ss, restStr)) restStr = "";

            // Parse endpoint IDs and shift them by one.
            double oDouble = 0.0, dDouble = 0.0;
            try {
                oDouble = stod(oStr);
                dDouble = stod(dStr);
            } catch (...) {
                // Skip malformed link rows.
                continue;
            }
            int oInt = static_cast<int>(oDouble) + 1;
            int dInt = static_cast<int>(dDouble) + 1;

            // Write the forward and reverse directed links.
            linkOut << oInt << "," << dInt;
            if (!restStr.empty()) {
                linkOut << "," << restStr;
            }
            linkOut << "\n";
            linkOut << dInt << "," << oInt;
            if (!restStr.empty()) {
                linkOut << "," << restStr;
            }
            linkOut << "\n";
        }

        linkIn.close();
        linkOut.close();
        cout << "Finish writing new_link.csv" << endl;
    }

    return 0;
}
