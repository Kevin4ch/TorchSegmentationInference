//
// Created by kevin4ch on 19-11-1.
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

        module = torch::jit::load(modelPath);
        kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        return true;
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return false;
    }

}

vector<SegObject> TorchSegEngine::segmentation(cv::Mat image) {
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255);
    auto img_tensor = torch::from_blob(img_float.data, {1, 3, inputSize.height, inputSize.width}, torch::kFloat32);
    for (int i = 0; i < 3; i++) {
        img_tensor[0][0] = img_tensor[0][0].sub_(meanValue[i]).div_(stdValue[i]);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(img_tensor);
    at::Tensor output_image = module.forward(inputs).toTensor();


    at::Tensor result = torch::argmax(output_image[0], 0);
    cv::Mat mat(inputSize, CV_8UC1, result.data_ptr());
    //模型正在训练，这个是标注。先用来测试后处理
    mat = cv::imread("/home/alien/CLionProjects/TorchSegmentation/test_result.png", -1);
    vector<SegObject> segObjects;
    for (int j = 1; j < cls_num; ++j) {
        cv::Mat cls_map;
        //to zero if px > threshold
        // 将高于本类别的像素位置的值置零,其他值不变
        cv::threshold(mat.clone(), cls_map, j, 0, CV_THRESH_TOZERO_INV);
        //to zero if px < threshold
        //将高于本类(由于threshold函数只判大于因此使用 j - 1)的像素位置的值不变,其他值置零
        cv::threshold(cls_map, cls_map, j - 1, 0, CV_THRESH_TOZERO);
        //to 255 if px ==threshold
        //将高于本类(j - 1)的像素值设置为255，其他值置零
        cv::threshold(cls_map, cls_map, j - 1, 255, CV_THRESH_BINARY);
        //腐蚀来清除部分噪声
        cv::erode(cls_map, cls_map, kernel);
        vector<vector<cv::Point>> contours;
        cv::findContours(cls_map, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); ++i) {
            vector<cv::Point> points = contours[i];
            cv::Point2f *vertices = new cv::Point2f[4];
            cv::RotatedRect rotatedRect = cv::minAreaRect(points);
            rotatedRect.points(vertices);
            SegObject obj;
            for (int k = 0; k < 4; ++k) {
                //模型输入的尺寸转回原图
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
    }
}
