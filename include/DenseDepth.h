#ifndef DENSEDEPTH_H
#define DENSEDEPTH_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

namespace ORB_SLAM3
{

    /**
     * Stereo dense depth generator using SGBM (Semi-Global Block Matching).
     *
     * Supports two modes based on Camera.type in the YAML settings:
     *   - "Rectified": images are already rectified (e.g. KITTI), skip rectification.
     *   - "PinHole":   raw images with distortion & stereo extrinsics (e.g. EuRoC),
     *                  performs stereo rectification + undistortion before SGBM.
     *
     * Usage:
     *   DenseDepth dd(settingsFile);
     *   cv::Mat depth = dd.Compute(imLeft, imRight);          // CV_32F depth in meters
     *   cv::Mat colored = DenseDepth::Colorize(depth, 80.0f); // false-color for visualization
     *   DenseDepth::Save(depth, "depth_0000.png");            // 16-bit millimeter PNG
     */
    class DenseDepth
    {
    public:
        /**
         * @brief Construct from a YAML settings file (same one used by System).
         *        Reads calibration, distortion, stereo extrinsics and auto-detects
         *        whether rectification is needed.
         */
        explicit DenseDepth(const std::string &settingsFile);

        /**
         * @brief Construct with explicit parameters (assumes already-rectified images).
         * @param fx         focal length in pixels
         * @param baseline   stereo baseline in meters
         * @param numDisp    number of disparities (must be divisible by 16)
         * @param blockSize  SGBM block size (odd, >= 1)
         */
        DenseDepth(float fx, float baseline, int numDisp = 128, int blockSize = 9);

        /**
         * @brief Compute a dense depth map from a stereo pair.
         *        If the camera type is not "Rectified", stereo rectification +
         *        undistortion is applied automatically before matching.
         * @param imLeft   left image (grayscale or BGR)
         * @param imRight  right image (grayscale or BGR)
         * @return CV_32F depth map in meters. Invalid pixels are 0.
         */
        cv::Mat Compute(const cv::Mat &imLeft, const cv::Mat &imRight);

        /**
         * @brief Save depth map as 16-bit PNG (values in millimeters).
         */
        static void Save(const cv::Mat &depth, const std::string &filename);

        /**
         * @brief Colorize depth for visualization.
         * @param maxDepth  depth range ceiling in meters for color mapping
         */
        static cv::Mat Colorize(const cv::Mat &depth, float maxDepth = 80.0f);

    private:
        void InitSGBM();
        void PrecomputeRectificationMaps(const cv::Mat &K1, const cv::Mat &D1,
                                         const cv::Mat &K2, const cv::Mat &D2,
                                         const cv::Mat &R, const cv::Mat &t,
                                         const cv::Size &imageSize);

        float mfx;       // focal length after rectification (pixels)
        float mBaseline; // stereo baseline (meters)
        int mNumDisp;    // number of disparities
        int mBlockSize;  // SGBM block size

        // Rectification state
        bool mbNeedRectify; // true if images must be rectified before matching
        cv::Mat mM1l, mM2l; // left  camera remap tables
        cv::Mat mM1r, mM2r; // right camera remap tables

        cv::Ptr<cv::StereoSGBM> mpSGBM;
    };

} // namespace ORB_SLAM3

#endif // DENSEDEPTH_H
