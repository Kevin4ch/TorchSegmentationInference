//
// Created by kevin on 19-11-1.
//

#include "TorchSegEngine.h"
#include <torch/script.h>

#include "torch/torch.h"


torch::jit::script::Module module;
cv::Mat kernel;

TorchSegEngine::TorchSegEngine(const string &modelPath, const cv::Size &size) {
    inputSize = size;
    loadModelFile(modelPath);
}

TorchSegEngine::TorchSegEngine(const vector<float> &mean, const vector<float> &std, const string &modelPath,
                               const cv::Size &size) {
    assert(mean.size() == 3 && std.size() == 3);
    inputSize = size;
    meanValue = mean;
    stdValue = std;
    loadModelFile(modelPath);
}

bool TorchSegEngine::loadModelFile(const string &modelPath) {
    try {
        cout<<"USE GPU:"<<torch::cuda::is_available()<<endl;
        module = torch::jit::load(modelPath);
        torch::Device device(torch::kCUDA);
        module.to(device);
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        return true;
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model"<<e.what()<<endl;
        return false;
    }

}

vector<SegObject> TorchSegEngine::segmentation(cv::Mat image) {
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    cv::resize(image,image,inputSize);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255.);
    auto img_tensor = torch::from_blob(img_float.data, {1,  inputSize.height, inputSize.width,3}, torch::kFloat32);
    for (int i = 0; i < 3; i++) {
        img_tensor[0][i] = img_tensor[0][i].sub_(meanValue[i]).div_(stdValue[i]);
    }
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
    img_tensor = img_tensor.to(torch::kCUDA);
    at::Tensor result = module.forward({img_tensor}).toTensor().to(torch::kCPU);
    result = result.argmax(1).squeeze().to(torch::kUInt8);
    cv::Mat mat(inputSize, CV_8UC1,result.data_ptr());
    vector<SegObject> segObjects;
    for (int j = 1; j < cls_num; ++j) {
        cv::Mat cls_map;
        //to zero if px > threshold
        cv::threshold(mat.clone(), cls_map, j, 0, CV_THRESH_TOZERO_INV);
        //to zero if px <= threshold
        cv::threshold(cls_map, cls_map, j - 1, 0, CV_THRESH_TOZERO);
        //to 255 if px ==threshold
        cv::threshold(cls_map, cls_map, j - 1, 255, CV_THRESH_BINARY);
        //filter smaller cell px area
        cv::erode(cls_map, cls_map, kernel);
        vector<vector<cv::Point>> contours;
        // RETR_EXTERNAL ignore  point inside of object
        cv::findContours(cls_map, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i) {
            vector<cv::Point> points = contours[i];
            //cout<<points.size()<<endl;
            cv::Point2f *vertices = new cv::Point2f[4];
            cv::RotatedRect rotatedRect = cv::minAreaRect(points);
            rotatedRect.points(vertices);
            SegObject obj;
            for (int k = 0; k < 4; ++k) {
                int imagePosX = (int) ((vertices[k].x / (float) mat.cols) * imageWidth);
                int imagePosy = (int) ((vertices[k].y / (float) mat.rows) * imageHeight);
                obj.points.push_back(cv::Point(imagePosX, imagePosy));
            }
            obj.clsType = j;

            delete vertices;
            segObjects.push_back(obj);
        }

    }
    return segObjects;
}

void TorchSegEngine::drawSegObjects(cv::Mat &src, vector<SegObject> &vertices, cv::Scalar color, int lineWidth) {
    for (int i = 0; i < vertices.size(); ++i) {
        SegObject obj = vertices[i];
        cv::line(src, obj.points[0], obj.points[1], color, lineWidth);
        cv::line(src, obj.points[1], obj.points[2], color, lineWidth);
        cv::line(src, obj.points[2], obj.points[3], color, lineWidth);
        cv::line(src, obj.points[3], obj.points[0], color, lineWidth);
        cv::putText(src,to_string(obj.clsType),obj.points[0],1,1,color);
    }
}
