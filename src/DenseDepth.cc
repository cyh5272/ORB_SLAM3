#include "DenseDepth.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM3
{

    // ---------------------------------------------------------------------------
    // Construct from YAML settings file
    // ---------------------------------------------------------------------------
    DenseDepth::DenseDepth(const std::string &settingsFile)
        : mbNeedRectify(false)
    {
        cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "[DenseDepth] Failed to open settings file: " << settingsFile << std::endl;
            mfx = 718.856f;
            mBaseline = 0.537f;
            mNumDisp = 128;
            mBlockSize = 9;
            InitSGBM();
            return;
        }

        // --- Read camera type ---
        std::string cameraType;
        fs["Camera.type"] >> cameraType;

        int imgW = (int)fs["Camera.width"];
        int imgH = (int)fs["Camera.height"];
        cv::Size imageSize(imgW, imgH);

        // --- Read left camera intrinsics ---
        float fx1 = fs["Camera1.fx"];
        float fy1 = fs["Camera1.fy"];
        float cx1 = fs["Camera1.cx"];
        float cy1 = fs["Camera1.cy"];
        cv::Mat K1 = (cv::Mat_<double>(3, 3) << fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1);

        if (cameraType == "Rectified")
        {
            // ----- Already rectified (e.g. KITTI) -----
            std::cout << "[DenseDepth] Camera type: Rectified – skipping stereo rectification" << std::endl;
            mfx = fx1;
            mBaseline = fs["Stereo.b"];
        }
        else
        {
            // ----- Need calibration + rectification (e.g. PinHole / EuRoC) -----
            std::cout << "[DenseDepth] Camera type: " << cameraType
                      << " – will perform stereo rectification" << std::endl;
            mbNeedRectify = true;

            // Left distortion
            float k1_1 = 0.f, k2_1 = 0.f, p1_1 = 0.f, p2_1 = 0.f;
            if (!fs["Camera1.k1"].empty())
                k1_1 = fs["Camera1.k1"];
            if (!fs["Camera1.k2"].empty())
                k2_1 = fs["Camera1.k2"];
            if (!fs["Camera1.p1"].empty())
                p1_1 = fs["Camera1.p1"];
            if (!fs["Camera1.p2"].empty())
                p2_1 = fs["Camera1.p2"];
            cv::Mat D1 = (cv::Mat_<double>(4, 1) << k1_1, k2_1, p1_1, p2_1);

            // Right camera intrinsics
            float fx2 = fs["Camera2.fx"];
            float fy2 = fs["Camera2.fy"];
            float cx2 = fs["Camera2.cx"];
            float cy2 = fs["Camera2.cy"];
            cv::Mat K2 = (cv::Mat_<double>(3, 3) << fx2, 0, cx2, 0, fy2, cy2, 0, 0, 1);

            // Right distortion
            float k1_2 = 0.f, k2_2 = 0.f, p1_2 = 0.f, p2_2 = 0.f;
            if (!fs["Camera2.k1"].empty())
                k1_2 = fs["Camera2.k1"];
            if (!fs["Camera2.k2"].empty())
                k2_2 = fs["Camera2.k2"];
            if (!fs["Camera2.p1"].empty())
                p1_2 = fs["Camera2.p1"];
            if (!fs["Camera2.p2"].empty())
                p2_2 = fs["Camera2.p2"];
            cv::Mat D2 = (cv::Mat_<double>(4, 1) << k1_2, k2_2, p1_2, p2_2);

            // Stereo extrinsics: Stereo.T_c1_c2 (4x4)
            cv::Mat Tc1c2;
            fs["Stereo.T_c1_c2"] >> Tc1c2;
            Tc1c2.convertTo(Tc1c2, CV_64F);

            // T_c1_c2 maps camera2 -> camera1. Extract R, t.
            cv::Mat R = Tc1c2.rowRange(0, 3).colRange(0, 3);
            cv::Mat t = Tc1c2.rowRange(0, 3).col(3);

            // Precompute remap tables & update mfx, mBaseline
            PrecomputeRectificationMaps(K1, D1, K2, D2, R, t, imageSize);
        }

        fs.release();

        mNumDisp = 128;
        mBlockSize = 9;
        InitSGBM();
    }

    // ---------------------------------------------------------------------------
    // Construct with explicit parameters (assumes already-rectified)
    // ---------------------------------------------------------------------------
    DenseDepth::DenseDepth(float fx, float baseline, int numDisp, int blockSize)
        : mfx(fx), mBaseline(baseline), mNumDisp(numDisp), mBlockSize(blockSize),
          mbNeedRectify(false)
    {
        InitSGBM();
    }

    // ---------------------------------------------------------------------------
    // Precompute stereo rectification remap look-up tables
    // ---------------------------------------------------------------------------
    void DenseDepth::PrecomputeRectificationMaps(
        const cv::Mat &K1, const cv::Mat &D1,
        const cv::Mat &K2, const cv::Mat &D2,
        const cv::Mat &R, const cv::Mat &t,
        const cv::Size &imageSize)
    {
        cv::Mat R1, R2, P1, P2, Q;
        cv::stereoRectify(K1, D1, K2, D2, imageSize,
                          R, t, R1, R2, P1, P2, Q,
                          cv::CALIB_ZERO_DISPARITY, -1, imageSize);

        cv::initUndistortRectifyMap(K1, D1, R1,
                                    P1.rowRange(0, 3).colRange(0, 3),
                                    imageSize, CV_32F, mM1l, mM2l);
        cv::initUndistortRectifyMap(K2, D2, R2,
                                    P2.rowRange(0, 3).colRange(0, 3),
                                    imageSize, CV_32F, mM1r, mM2r);

        // After rectification the new focal length is P1[0,0]
        mfx = static_cast<float>(P1.at<double>(0, 0));
        // Baseline from Q matrix: b = -1/Q[3][2]  or  T[0]/norm
        mBaseline = static_cast<float>(std::abs(P2.at<double>(0, 3) / P2.at<double>(0, 0)));

        std::cout << "[DenseDepth] Rectified fx=" << mfx
                  << " baseline=" << mBaseline << " m" << std::endl;
    }

    // ---------------------------------------------------------------------------
    // Create SGBM matcher
    // ---------------------------------------------------------------------------
    void DenseDepth::InitSGBM()
    {
        // Ensure numDisp is a multiple of 16
        mNumDisp = std::max(16, (mNumDisp / 16) * 16);

        int P1 = 8 * mBlockSize * mBlockSize;
        int P2 = 32 * mBlockSize * mBlockSize;

        mpSGBM = cv::StereoSGBM::create(
            0,                             // minDisparity
            mNumDisp,                      // numDisparities
            mBlockSize,                    // blockSize
            P1,                            // P1 – penalty on disparity changes by +/-1
            P2,                            // P2 – penalty on disparity changes by more than 1
            1,                             // disp12MaxDiff  (left-right consistency check)
            63,                            // preFilterCap
            10,                            // uniquenessRatio
            100,                           // speckleWindowSize
            32,                            // speckleRange
            cv::StereoSGBM::MODE_SGBM_3WAY // faster 3-way DP
        );
    }

    // ---------------------------------------------------------------------------
    // Compute dense depth
    // ---------------------------------------------------------------------------
    cv::Mat DenseDepth::Compute(const cv::Mat &imLeft, const cv::Mat &imRight)
    {
        cv::Mat procL, procR;

        // ---- Step 1: Stereo rectification + undistortion (if needed) ----
        if (mbNeedRectify)
        {
            cv::remap(imLeft, procL, mM1l, mM2l, cv::INTER_LINEAR);
            cv::remap(imRight, procR, mM1r, mM2r, cv::INTER_LINEAR);
        }
        else
        {
            procL = imLeft;
            procR = imRight;
        }

        // ---- Step 2: Convert to grayscale ----
        cv::Mat grayL, grayR;
        if (procL.channels() == 3)
            cv::cvtColor(procL, grayL, cv::COLOR_BGR2GRAY);
        else
            grayL = procL;

        if (procR.channels() == 3)
            cv::cvtColor(procR, grayR, cv::COLOR_BGR2GRAY);
        else
            grayR = procR;

        // Compute disparity (result is CV_16S, units = disparity * 16)
        cv::Mat disparity16S;
        mpSGBM->compute(grayL, grayR, disparity16S);

        // Convert to float disparity
        cv::Mat disparityF;
        disparity16S.convertTo(disparityF, CV_32F, 1.0 / 16.0);

        // Convert disparity to depth: depth = fx * baseline / disparity
        cv::Mat depth = cv::Mat::zeros(disparityF.size(), CV_32F);
        float fb = mfx * mBaseline;

        for (int r = 0; r < disparityF.rows; ++r)
        {
            const float *pDisp = disparityF.ptr<float>(r);
            float *pDepth = depth.ptr<float>(r);
            for (int c = 0; c < disparityF.cols; ++c)
            {
                float d = pDisp[c];
                if (d > 0.5f) // valid disparity threshold
                    pDepth[c] = fb / d;
                // else remains 0 (invalid)
            }
        }

        return depth;
    }

    // ---------------------------------------------------------------------------
    // Save depth as 16-bit PNG (millimeters)
    // ---------------------------------------------------------------------------
    void DenseDepth::Save(const cv::Mat &depth, const std::string &filename)
    {
        cv::Mat depth16U;
        // Convert meters -> millimeters, clamp to uint16 range
        depth.convertTo(depth16U, CV_16U, 1000.0, 0.0);
        cv::imwrite(filename, depth16U);
    }

    // ---------------------------------------------------------------------------
    // Colorize depth for display
    // ---------------------------------------------------------------------------
    cv::Mat DenseDepth::Colorize(const cv::Mat &depth, float maxDepth)
    {
        cv::Mat normalized;
        depth.convertTo(normalized, CV_8U, 255.0 / maxDepth, 0.0);

        cv::Mat colored;
        cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);

        // Mark invalid pixels (depth==0) as black
        for (int r = 0; r < depth.rows; ++r)
        {
            const float *pD = depth.ptr<float>(r);
            cv::Vec3b *pC = colored.ptr<cv::Vec3b>(r);
            for (int c = 0; c < depth.cols; ++c)
            {
                if (pD[c] <= 0.f)
                    pC[c] = cv::Vec3b(0, 0, 0);
            }
        }

        return colored;
    }

} // namespace ORB_SLAM3
