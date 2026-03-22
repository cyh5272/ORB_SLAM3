/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{  
    if(argc < 5)
    {
        cerr << endl << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        cerr << endl << "Notes: \n - You can pass one or more pairs of <image_folder> <times_file>.\n - times_file lines can be either a single timestamp (nanoseconds or seconds) OR two columns (timestamp image_filename) or (image_filename timestamp).\n - If timestamps are integer nanoseconds, they will be converted to seconds.\n - Optional last argument when odd number of trailing args is treated as output trajectory file name." << endl;
        return 1;
    }

    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
    cout << "Loading images for sequence " << seq << "...";
    // Use the image folder exactly as provided on the command line. Do not append EuRoC-specific subpaths.
    LoadImages(string(argv[(2*seq)+3]), string(argv[(2*seq)+4]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);


    int fps = 20;
    float dT = 1.f/fps;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, false);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        cv::Mat im;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_UNCHANGED); //,CV_LOAD_IMAGE_UNCHANGED);
            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            if(imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            // cout << "tframe = " << tframe << endl;
            SLAM.TrackMonocular(im,tframe); // TODO change to monocular_inertial

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            //std::cout << "T: " << T << std::endl;
            //std::cout << "ttrack: " << ttrack << std::endl;

            if(ttrack<T) {
                //std::cout << "usleep: " << (dT-ttrack) << std::endl;
                usleep((T-ttrack)*1e6); // 1e6
            }
        }

        if(seq < num_seq - 1)
        {
            string kf_file_submap =  "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
            string f_file_submap =  "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file_submap);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }

    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes(strPathTimes.c_str());
    if(!fTimes.is_open())
    {
        cerr << "Failed opening times file: " << strPathTimes << endl;
        return;
    }

    vTimeStamps.clear();
    vstrImages.clear();
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    string line;
    while(std::getline(fTimes, line))
    {
        if(line.empty()) continue;

        // Trim leading/trailing spaces
        auto l = line.find_first_not_of(" \t\r\n");
        if(l == string::npos) continue;
        auto r = line.find_last_not_of(" \t\r\n");
        string s = line.substr(l, r - l + 1);

        stringstream ss(s);
        vector<string> tokens;
        string token;
        while(ss >> token) tokens.push_back(token);

        double timestamp = 0.0;
        string imageName;

        if(tokens.size() == 1)
        {
            // Single column: either a timestamp (common in Euroc/TUM) or a filename
            // Decide by checking if token is numeric (contains digits only)
            bool isNumber = true;
            for(char c: tokens[0]) if(!(isdigit(c) || c=='.' || c=='-' )) { isNumber = false; break; }
            if(isNumber)
            {
                // Parse as timestamp. Could be nanoseconds (large integer) or seconds (floating)
                try {
                    if(tokens[0].find('.') == string::npos && tokens[0].size() > 10)
                    {
                        // integer nanoseconds
                        long long tns = stoll(tokens[0]);
                        timestamp = double(tns) * 1e-9;
                    }
                    else
                    {
                        timestamp = stod(tokens[0]);
                    }
                } catch(...) { timestamp = 0.0; }
                // derive image filename from timestamp (common pattern: <timestamp>.png)
                // But it's safer to assume files are named with the timestamp integer part
                // We'll try both integer (ns) and seconds-based naming to be flexible.
                // Prefer integer nanosecond name if exists.
                // Compose possible names
                string name_ns, name_s;
                {
                    // integer nanoseconds string from original token if integer
                    bool allDigits = true;
                    for(char c: tokens[0]) if(!isdigit(c)) { allDigits = false; break; }
                    if(allDigits)
                    {
                        name_ns = tokens[0] + ".png";
                    }
                    // seconds-based
                    ostringstream oss;
                    oss.precision(9);
                    oss << fixed << timestamp;
                    name_s = oss.str() + ".png";
                }

                // Check which file exists
                string full_ns = strImagePath + "/" + name_ns;
                string full_s = strImagePath + "/" + name_s;
                // We will prefer ns name if file exists, else seconds name, else try padded integer names
                if(!name_ns.empty()){
                    if(FILE *f = fopen(full_ns.c_str(), "r")) { fclose(f); imageName = name_ns; }
                }
                if(imageName.empty()){
                    if(FILE *f = fopen(full_s.c_str(), "r")) { fclose(f); imageName = name_s; }
                }
                if(imageName.empty()){
                    // fallback: use token as filename (maybe already has extension)
                    imageName = tokens[0];
                }
            }
            else
            {
                // token is not numeric -> image filename
                imageName = tokens[0];
                timestamp = 0.0;
            }
        }
        else if(tokens.size() >= 2)
        {
            // Two columns: could be (timestamp filename) or (filename timestamp)
            // Detect which token is numeric
            bool firstIsNum = true;
            for(char c: tokens[0]) if(!(isdigit(c) || c=='.' || c=='-' )) { firstIsNum = false; break; }
            bool secondIsNum = true;
            for(char c: tokens[1]) if(!(isdigit(c) || c=='.' || c=='-' )) { secondIsNum = false; break; }

            if(firstIsNum && !secondIsNum)
            {
                // timestamp filename
                string ts = tokens[0];
                try {
                    if(ts.find('.') == string::npos && ts.size() > 10)
                    {
                        long long tns = stoll(ts);
                        timestamp = double(tns) * 1e-9;
                    }
                    else timestamp = stod(ts);
                } catch(...) { timestamp = 0.0; }
                imageName = tokens[1];
            }
            else if(!firstIsNum && secondIsNum)
            {
                imageName = tokens[0];
                string ts = tokens[1];
                try {
                    if(ts.find('.') == string::npos && ts.size() > 10)
                    {
                        long long tns = stoll(ts);
                        timestamp = double(tns) * 1e-9;
                    }
                    else timestamp = stod(ts);
                } catch(...) { timestamp = 0.0; }
            }
            else
            {
                // both numeric or both non-numeric: assume first is timestamp, second is filename
                try {
                    if(tokens[0].find('.') == string::npos && tokens[0].size() > 10)
                    {
                        long long tns = stoll(tokens[0]);
                        timestamp = double(tns) * 1e-9;
                    }
                    else timestamp = stod(tokens[0]);
                } catch(...) { timestamp = 0.0; }
                imageName = tokens[1];
            }
        }

        // Normalize and push
        string fullImagePath;
        // If imageName already contains a path separator, treat as full/relative path
        if(!imageName.empty() && (imageName.find('/') != string::npos || imageName.find('\\') != string::npos))
            fullImagePath = imageName;
        else
            fullImagePath = strImagePath + "/" + imageName;

        vstrImages.push_back(fullImagePath);
        vTimeStamps.push_back(timestamp);
    }
}
