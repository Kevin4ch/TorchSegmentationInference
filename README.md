### Torch gpu inference （segmentation） in c++

```
int main() {
    TorchSegEngine engine("/home/alien/CLionProjects/TorchSegmentation/test.pt",cv::Size(512,512));
    cv::Mat mat = cv::imread("/home/alien/CLionProjects/TorchSegmentation/20190403_0000026513.jpg");
    vector<SegObject> obj = engine.segmentation(mat.clone());
    engine.drawSegObjects(mat,obj,cv::Scalar(0,2,255),1);
    cv::imshow("",mat);
    cv::waitKey(0);
    return 0;
}
```
