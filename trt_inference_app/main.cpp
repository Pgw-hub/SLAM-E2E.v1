#include <iostream>
#include "etoeNet.h"
#include "filesystem.h"
#include "opencv2/opencv.hpp"

using namespace std;

int main(int argc, char **argv){

    etoeNet etoe_net;

    std::string onnx_file_path = "/home/parkgeonwoo/Documents/e2e/JoyAI_FINAL/src/model-60.simplified.onnx";

    //onnx file을 불러서 tensorrt 초기화
    etoe_net.loadOnnxFile(onnx_file_path);

    // TODO : inference할 image를 listup
    std::string data_dir = "/home/parkgeonwoo/Documents/e2e/JoyAI_FINAL/DATA/lab_front/data6";
    std::vector<std::string> img_extensions = {"jpg","png"};

    std::vector<std::string> file_list;
    listDir(data_dir, file_list, FILE_REGULAR);

    // int i = 1;

    std::vector<std::string> img_list;
    for(auto &file_path : file_list){
        if(fileHasExtension(file_path,img_extensions)){
            img_list.push_back(file_path);
        }
    }

    int i = 1;

    // TODO :  inference loop
    for(auto &img_path :img_list){
        cv::Mat img_mat = cv::imread(img_path);
        etoe_net.runInference(img_mat);
        cv::imshow("img",img_mat);
        cv::waitKey(200);
    }

    return 0;
}