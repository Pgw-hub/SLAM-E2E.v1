#ifndef __ETOE_NET__
#define __ETOE_NET__

#include <opencv2/opencv.hpp>
#include "tensorNet.h"

class etoeNet : tensorNet
{
public:
    etoeNet();
    ~etoeNet();

    void loadOnnxFile(const std::string &onnx_file_path);

    void runInference(const cv::Mat &img_mat);

private:
    cv::Mat m_img_crpped_rgb_f_mat;
};

#endif