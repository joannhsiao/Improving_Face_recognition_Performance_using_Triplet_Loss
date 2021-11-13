#include <list>
#include <iostream>
// #include <time.h>
#include <fstream>
#include <cstring>
#include <cmath>
//#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mxnet/c_predict_api.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
using namespace std;
using namespace cv;

#include "LFQueue.hpp"
#include "MTCNN.hpp"
#include "Feature.hpp"
#include "CameraSettings.h"
#include "clustering.h"
