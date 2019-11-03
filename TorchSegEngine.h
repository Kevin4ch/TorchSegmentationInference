//
// Created by kevin on 19-11-1.
//

#ifndef TORCHSEGMENTATION_TORCHSEGENGINE_H
#define TORCHSEGMENTATION_TORCHSEGENGINE_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

struct SegObject{
    vector<cv::Point> points;
    int clsType = 0;
};

class TorchSegEngine {
public:
    TorchSegEngine(const string &modelPath,const cv::Size &inputSize);

    TorchSegEngine(const vector<float> &mean,const  vector<float> &std, const string &modelPath,const cv::Size &size);

    vector<SegObject> segmentation(cv::Mat image);

    void drawSegObjects(cv::Mat &src,vector<SegObject> &vertices, cv::Scalar color, int lineWidth);
private:
    std::vector<float> meanValue = {0.485, 0.456, 0.406};
    std::vector<float> stdValue = {0.229, 0.224, 0.225};
    int cls_num = 13;
    cv::Size inputSize;
    bool loadModelFile(const string &modelPath);


};


#endif //TORCHSEGMENTATION_TORCHSEGENGINE_H
