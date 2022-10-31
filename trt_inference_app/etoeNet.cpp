#include "etoeNet.h"

etoeNet::etoeNet() : tensorNet()
{

}

etoeNet::~etoeNet()
{

}

void etoeNet::loadOnnxFile(const std::string &onnx_file_path){
    std::vector<std::string> input_blobs = {"input"};
    std::vector<std::string> output_blobs = {"output"};
    LoadNetwork(NULL, onnx_file_path.c_str(), NULL, input_blobs, output_blobs, 1, TYPE_FP32);

    m_img_crpped_rgb_f_mat = cv::Mat(cv::Size(320,70), CV_32FC3, mInputs[0].CPU);
}

void etoeNet::runInference(const cv::Mat &img_mat){
        
        //------------------------------------------------
        //preprocessing
        cv::Mat img_resized_mat;
        cv::resize(img_mat, img_resized_mat, cv::Size(320,160));
        cv::Mat img_crpped_mat = img_resized_mat(cv::Rect(0,65,320,70));
        
        cv::Mat img_crpped_rgb_mat;
        cv::cvtColor(img_crpped_mat, img_crpped_rgb_mat, cv::COLOR_BGR2RGB);
        img_crpped_rgb_mat.convertTo(m_img_crpped_rgb_f_mat,CV_32FC3, (1.0 / 255.0), -1.0);
        //---------------------------------------------------
        
        ProcessNetwork(true);

        std::cout << "run" << std::endl;

        //TODO :: inference 결과를 가져오기
        float network_output = *(mOutputs[0].CPU);
        float steering_angle;

        if(network_output < -0.75){
            steering_angle = -1.00;
        }
        else if(network_output < -0.25){
            steering_angle = -0.5;
        }
        else if(network_output < 0.25){
            steering_angle = 0;
        }else if(network_output < 0.75){
            steering_angle = 0.5;
        } 
        else{
            steering_angle = 1.00;
        }

        steering_angle *= 20.0;

        std::cout << "actual steering angle"  << steering_angle << std::endl;
}