// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//include dlib library
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>

//additional functions
#include "FileProc.hpp"
#include "FileProcDNN.hpp"
#include "TransformLandmark.hpp"

#define toShowTimeLog 0

using namespace std;
using namespace cv;

double Load_model_time = 0;

// Parameters
int dev_type = 1;  // 1: cpu, 2: gpu
int dev_id = 1;  // arbitrary.
mx_uint num_input_nodes = 1;  // 1 for feedforward
const char* input_key[1] = {"data"};
const char** input_keys = input_key;

class BoundingBox{
public:
  mx_float Reg_nms_x1;
  mx_float Reg_nms_y1;
  mx_float Reg_nms_x2;
  mx_float Reg_nms_y2;
  mx_float score;
  mx_float x1;
  mx_float y1;
  mx_float x2;
  mx_float y2;
  mx_float area;
};

class RefineBox{
public:
  mx_float x1;
  mx_float y1;
  mx_float x2;
  mx_float y2;
  mx_float score;
  mx_float area;
};

class LMK_Point{
public:
  double LMK[10];
};


class PreLoadPNetPool
{
public:
  std::vector<PredictorHandle> md_PNet;
  std::vector<double> scales;

  BufferFile PNet_json_data;
  BufferFile PNet_param_data;
  int imgH;
  int imgW;

  PreLoadPNetPool(std::string PNet_json_file, std::string PNet_param_file):
    imgH(0), imgW(0), PNet_json_data(PNet_json_file), PNet_param_data(PNet_param_file)
  {
    scales.clear();
  }
  ~PreLoadPNetPool(){release();}

  void reload(int H, int W, double min_s = 0)
  {
    if(H == imgH && W == imgW)
      return;

    release();
    load(H,W,min_s);
  }
  void load(int H, int W, double min_face_size = 0)
  {
    dev_type = 2;
    dev_id =0;
    imgH = H;
    imgW = W;

    int MIN_DET_SIZE = 12;
    double minl = min(H, W);
    double m = MIN_DET_SIZE / 20.;
    minl *= m;

    int drop_scale = 0;
    min_face_size *= m;
    // To let the face that small than the min_face_size can't be detected by PNet
    // Just only not to let PNet to predect the image lager than some scales.
    // Because in the PNet predector, it's change image size rather then change scaler box.
    while(min_face_size > MIN_DET_SIZE)
    {
      min_face_size *= 0.709;
      drop_scale++;
    }

    while(minl > MIN_DET_SIZE)
    {
      double a_scale = m;
      m *=0.709;
      minl *= 0.709;
      if(drop_scale)
      {
        drop_scale--;
        continue;
      }
      scales.push_back(a_scale);

      // prepare PredictorHandle
      int width = W * a_scale;
      int height = H * a_scale;
      int channels = 3;
      const mx_uint input_shape_indptr[2] = { 0, 4 };
      const mx_uint input_shape_data[4] = { 1, static_cast<mx_uint>(channels), static_cast<mx_uint>(height), static_cast<mx_uint>(width) };

      PredictorHandle pred_hnd = 0;
      MXPredCreate((const char*)PNet_json_data.GetBuffer(),
                   (const char*)PNet_param_data.GetBuffer(),
                   static_cast<size_t>(PNet_param_data.GetLength()),
                   dev_type,
                   dev_id,
                   num_input_nodes,
                   input_keys,
                   input_shape_indptr,
                   input_shape_data,
                   &pred_hnd);
      assert(pred_hnd);
      md_PNet.push_back(pred_hnd);
    }
  }
  void release()
  {
    int ph_size = md_PNet.size();
    for(int i=0;i<ph_size;i++)
      MXPredFree(md_PNet[i]);
    md_PNet.clear();
    scales.clear();
  }
};

vector<size_t> sort_indexes(const vector<RefineBox> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1].score < v[i2].score;});

  return idx;
}

vector<int> nms(vector<RefineBox> Refine_Box, float overlap_threshold, int model = 0) {
  //model: 0 = Union, 1 = Min
  std::vector<int> Pick_Bounding_Box;
  vector<size_t> index = sort_indexes(Refine_Box);

  while(index.size() > 0){
    int last = index.size() - 1;
    int i = index[last];
    Pick_Bounding_Box.push_back(i);

    std::vector<mx_float> overlap(last);

    for(int j = 0; j < last; j++){
      mx_float temp_x1, temp_y1, temp_x2, temp_y2;

      temp_x1 = (Refine_Box[index[j]].x1 >  Refine_Box[i].x1) ? Refine_Box[index[j]].x1 : Refine_Box[i].x1;
      temp_y1 = (Refine_Box[index[j]].y1 >  Refine_Box[i].y1) ? Refine_Box[index[j]].y1 : Refine_Box[i].y1;
      temp_x2 = (Refine_Box[index[j]].x2 >  Refine_Box[i].x2) ? Refine_Box[i].x2 : Refine_Box[index[j]].x2;
      temp_y2 = (Refine_Box[index[j]].y2 >  Refine_Box[i].y2) ? Refine_Box[i].y2 : Refine_Box[index[j]].y2;

      mx_float temp_w = (0. > (temp_x2 - temp_x1 + 1.)) ? 0. : (temp_x2 - temp_x1 + 1.);
      mx_float temp_h = (0. > (temp_y2 - temp_y1 + 1.)) ? 0. : (temp_y2 - temp_y1 + 1.);
      mx_float inter = (temp_w * temp_h);
      if(model == 0)
        overlap[j] = inter / (Refine_Box[i].area + Refine_Box[index[j]].area - inter);
      else{
        mx_float minarea = (Refine_Box[i].area > Refine_Box[index[j]].area) ? Refine_Box[index[j]].area : Refine_Box[i].area;
        overlap[j] = inter / minarea;
      }

    }

    index.erase(index.begin() + last);
    for(int j = overlap.size() - 1; j >= 0; j--){
      if(overlap[j] > overlap_threshold){
        index.erase(index.begin() + j);
      }
    }
  }

  return Pick_Bounding_Box;
}

vector<RefineBox> convert_to_square(vector<RefineBox> Refine_Box){
  mx_float temp_w, temp_h, temp_max;

  #pragma omp parallel for
  for(unsigned int i = 0; i < Refine_Box.size(); i++){
    temp_w = Refine_Box[i].x2 - Refine_Box[i].x1 + 1;
    temp_h = Refine_Box[i].y2 - Refine_Box[i].y1 + 1;
    temp_max = (temp_w > temp_h) ? temp_w : temp_h;

    Refine_Box[i].x1 = Refine_Box[i].x1 + temp_w * 0.5 - temp_max * 0.5;
    Refine_Box[i].y1 = Refine_Box[i].y1 + temp_h * 0.5 - temp_max * 0.5;
    Refine_Box[i].x2 = Refine_Box[i].x1 + temp_max - 1;
    Refine_Box[i].y2 = Refine_Box[i].y1 + temp_max - 1;

    Refine_Box[i].x1 = round(Refine_Box[i].x1);
    Refine_Box[i].y1 = round(Refine_Box[i].y1);
    Refine_Box[i].x2 = round(Refine_Box[i].x2);
    Refine_Box[i].y2 = round(Refine_Box[i].y2);
    Refine_Box[i].area = (Refine_Box[i].x2 - Refine_Box[i].x1 + 1) * (Refine_Box[i].y2 - Refine_Box[i].y1 + 1);
  }

  return Refine_Box;
}

int PNet_detector(Mat img, double scale, double H, double W, BufferFile& PNet_json_data, BufferFile& PNet_param_data, vector<BoundingBox>& Bounding_Box){
  // Image size and channels
  int width = W * scale;
  int height = H * scale;
  int channels = 3;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1, static_cast<mx_uint>(channels), static_cast<mx_uint>(height), static_cast<mx_uint>(width) };
  PredictorHandle pred_hnd = 0;

  // Create Predictor
  time_t start = clock();

  MXPredCreate((const char*)PNet_json_data.GetBuffer(),
               (const char*)PNet_param_data.GetBuffer(),
               static_cast<size_t>(PNet_param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);

  time_t end = clock();
  Load_model_time += double(end - start);

  assert(pred_hnd);

  int image_size = width * height * channels;

  // Read Image Data
  std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

  GetImageFile(img, image_data.data(), channels, cv::Size(width, height));

  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), image_size);

  // Do Predict Forward
  MXPredForward(pred_hnd);

  mx_uint output_index = 0; // 0:bounding box 1: score
  mx_uint *shape0 = 0;
  mx_uint shape_len0;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape0, &shape_len0);

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len0; ++i) {
    size *= shape0[i];
  }

  std::vector<mx_float> data0(size);
  MXPredGetOutput(pred_hnd, output_index, &(data0[0]), size);

  std::vector<mx_float> x1, y1, x2, y2;

  for(unsigned int i = 0; i < shape0[2] * shape0[3]; i++){
    x1.push_back(data0[i]);
    y1.push_back(data0[i + shape0[2] * shape0[3]]);
    x2.push_back(data0[i + (shape0[2] * shape0[3]) * 2]);
    y2.push_back(data0[i + (shape0[2] * shape0[3]) * 3]);
  }

  output_index = 1; // 0:bounding box 1: score
  mx_uint *shape1 = 0;
  mx_uint shape_len1;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape1, &shape_len1);

  size = 1;
  for (mx_uint i = 0; i < shape_len1; ++i) \
    size *= shape1[i];

  std::vector<mx_float> data1(size);
  MXPredGetOutput(pred_hnd, output_index, &(data1[0]), size);

  std::vector<mx_float> score;
  // Save Output Data
  for(unsigned int i = size - shape1[2] * shape1[3] ; i < size ; i++)
    score.push_back(mx_float(data1[i]));

  float PNet_threshold = 0.6;
  double PNet_stride = 2;
  double PNet_cellsize = 12;

  for(unsigned int i = 0; i < score.size(); i++){
    if(score[i] > PNet_threshold){

      BoundingBox Box;
      Box.Reg_nms_x1 = round((PNet_stride * int(i % shape1[3]) + 1)/scale);
      Box.Reg_nms_y1 = round((PNet_stride * int(i / shape1[3]) + 1)/scale);
      Box.Reg_nms_x2 = round((PNet_stride * int(i % shape1[3]) + 1 + PNet_cellsize)/scale);
      Box.Reg_nms_y2 = round((PNet_stride * int(i / shape1[3]) + 1 + PNet_cellsize)/scale);
      Box.score = score[i];
      Box.x1 = x1[i];
      Box.y1 = y1[i];
      Box.x2 = x2[i];
      Box.y2 = y2[i];
      Box.area = (Box.Reg_nms_x2 - Box.Reg_nms_x1 + 1) * (Box.Reg_nms_y2 - Box.Reg_nms_y1 + 1);
      Bounding_Box.push_back(Box);
    }
  }

  // Release Predictor
  MXPredFree(pred_hnd);

  return 0;
}


int PNet_detector(Mat img, double scale, double H, double W, PredictorHandle& pred_hnd, vector<BoundingBox>& Bounding_Box){
  // Image size and channels
  int width = W * scale;
  int height = H * scale;
  int channels = 3;
  int image_size = width * height * channels;

  // Read Image Data
  std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

  GetImageFile(img, image_data.data(), channels, cv::Size(width, height));

  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), image_size);

  // Do Predict Forward
  MXPredForward(pred_hnd);

  mx_uint output_index = 0; // 0:bounding box 1: score
  mx_uint *shape0 = 0;
  mx_uint shape_len0;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape0, &shape_len0);

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len0; ++i) {
    size *= shape0[i];
  }

  std::vector<mx_float> data0(size);
  MXPredGetOutput(pred_hnd, output_index, &(data0[0]), size);

  std::vector<mx_float> x1, y1, x2, y2;

  for(unsigned int i = 0; i < shape0[2] * shape0[3]; i++){
    x1.push_back(data0[i]);
    y1.push_back(data0[i + shape0[2] * shape0[3]]);
    x2.push_back(data0[i + (shape0[2] * shape0[3]) * 2]);
    y2.push_back(data0[i + (shape0[2] * shape0[3]) * 3]);
  }

  output_index = 1; // 0:bounding box 1: score
  mx_uint *shape1 = 0;
  mx_uint shape_len1;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape1, &shape_len1);

  size = 1;
  for (mx_uint i = 0; i < shape_len1; ++i) \
    size *= shape1[i];

  std::vector<mx_float> data1(size);
  MXPredGetOutput(pred_hnd, output_index, &(data1[0]), size);

  std::vector<mx_float> score;
  // Save Output Data
  for(unsigned int i = size - shape1[2] * shape1[3] ; i < size ; i++)
    score.push_back(mx_float(data1[i]));

  float PNet_threshold = 0.6;
  double PNet_stride = 2;
  double PNet_cellsize = 12;

  for(unsigned int i = 0; i < score.size(); i++){
    if(score[i] > PNet_threshold){

      BoundingBox Box;
      Box.Reg_nms_x1 = round((PNet_stride * int(i % shape1[3]) + 1)/scale);
      Box.Reg_nms_y1 = round((PNet_stride * int(i / shape1[3]) + 1)/scale);
      Box.Reg_nms_x2 = round((PNet_stride * int(i % shape1[3]) + 1 + PNet_cellsize)/scale);
      Box.Reg_nms_y2 = round((PNet_stride * int(i / shape1[3]) + 1 + PNet_cellsize)/scale);
      Box.score = score[i];
      Box.x1 = x1[i];
      Box.y1 = y1[i];
      Box.x2 = x2[i];
      Box.y2 = y2[i];
      Box.area = (Box.Reg_nms_x2 - Box.Reg_nms_x1 + 1) * (Box.Reg_nms_y2 - Box.Reg_nms_y1 + 1);
      Bounding_Box.push_back(Box);
    }
  }
  return 0;
}

int RNet_detector(vector<cv::Mat> IMGDATA, BufferFile& RNet_json_data, BufferFile& RNet_param_data, vector<mx_float>& score, vector<mx_float>& x1, vector<mx_float>&  y1, vector<mx_float>&  x2, vector<mx_float>&  y2){
  // Image size and channels
  int width = 24;
  int height = 24;
  int channels = 3;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { static_cast<mx_uint>(IMGDATA.size()), static_cast<mx_uint>(channels), static_cast<mx_uint>(height), static_cast<mx_uint>(width) };
  PredictorHandle pred_hnd = 0;

  // Create Predictor
  time_t start = clock();

  MXPredCreate((const char*)RNet_json_data.GetBuffer(),
               (const char*)RNet_param_data.GetBuffer(),
               static_cast<size_t>(RNet_param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);

  time_t end = clock();
  Load_model_time += double(end - start);

  assert(pred_hnd);
  int image_size = width * height * channels;

  // Read Image Data
  std::vector<mx_float> image_data = std::vector<mx_float>(IMGDATA.size() * image_size);

  CombineIMGDATA(IMGDATA, image_data.data(), channels, cv::Size(width, height));

  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), IMGDATA.size() * image_size);

  // Do Predict Forward
  MXPredForward(pred_hnd);

  mx_uint output_index = 0; // 0:bounding box 1: score
  mx_uint *shape0 = 0;
  mx_uint shape_len0;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape0, &shape_len0);

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len0; ++i) {
    size *= shape0[i];
  }

  std::vector<mx_float> data0(size);
  MXPredGetOutput(pred_hnd, output_index, &(data0[0]), size);

  for(unsigned int i = 0; i < IMGDATA.size(); i++){
    x1.push_back(data0[i * 4]);
    y1.push_back(data0[i * 4 + 1]);
    x2.push_back(data0[i * 4 + 2]);
    y2.push_back(data0[i * 4 + 3]);
  }

  output_index = 1; // 0:bounding box 1: score
  mx_uint *shape1 = 0;
  mx_uint shape_len1;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape1, &shape_len1);

  size = 1;
  for (mx_uint i = 0; i < shape_len1; ++i) \
    size *= shape1[i];

  std::vector<mx_float> data1(size);
  MXPredGetOutput(pred_hnd, output_index, &(data1[0]), size);

  // Save Output Data
  for(unsigned int i = 0; i < size ; i++){
    if(i % 2 == 1){
      score.push_back(mx_float(data1[i]));
    }
  }
  // Release Predictor
  MXPredFree(pred_hnd);

  return 0;
}

int ONet_detector(vector<cv::Mat> IMGDATA, BufferFile& ONet_json_data, BufferFile& ONet_param_data, vector<mx_float>& score, vector<mx_float>& x1, vector<mx_float>&  y1, vector<mx_float>&  x2, vector<mx_float>&  y2, vector<LMK_Point>& LMK){
  // Image size and channels
  int width = 48;
  int height = 48;
  int channels = 3;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { static_cast<mx_uint>(IMGDATA.size()), static_cast<mx_uint>(channels), static_cast<mx_uint>(height), static_cast<mx_uint>(width) };
  PredictorHandle pred_hnd = 0;

  // Create Predictor
  time_t start = clock();
  MXPredCreate((const char*)ONet_json_data.GetBuffer(),
               (const char*)ONet_param_data.GetBuffer(),
               static_cast<size_t>(ONet_param_data.GetLength()),
               dev_type,
               dev_id,
               num_input_nodes,
               input_keys,
               input_shape_indptr,
               input_shape_data,
               &pred_hnd);
  time_t end = clock();
  Load_model_time += double(end - start);
  assert(pred_hnd);
  unsigned int image_size = width * height * channels;
  unsigned int DATA_size = IMGDATA.size();

  // Read Image Data
  std::vector<mx_float> image_data = std::vector<mx_float>(DATA_size * image_size);

  CombineIMGDATA(IMGDATA, image_data.data(), channels, cv::Size(width, height));

  // Set Input Image
  MXPredSetInput(pred_hnd, "data", image_data.data(), DATA_size * image_size);

  // Do Predict Forward
  MXPredForward(pred_hnd);

  mx_uint output_index = 0; // 0:bounding box 1: score
  mx_uint *shape0 = 0;
  mx_uint shape_len0;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape0, &shape_len0);

  size_t size = 1;
  for (mx_uint i = 0; i < shape_len0; ++i) {
    size *= shape0[i];
  }

  std::vector<mx_float> data0(size);
  MXPredGetOutput(pred_hnd, output_index, &(data0[0]), size);

  LMK.resize(DATA_size);

  #pragma omp parallel for
  for(unsigned int i = 0; i < DATA_size; i++){
    LMK_Point temp_LMK;
    temp_LMK.LMK[0] = data0[i * 10];
    temp_LMK.LMK[1] = data0[i * 10 + 5];
    temp_LMK.LMK[2] = data0[i * 10 + 1];
    temp_LMK.LMK[3] = data0[i * 10 + 6];
    temp_LMK.LMK[4] = data0[i * 10 + 2];
    temp_LMK.LMK[5] = data0[i * 10 + 7];
    temp_LMK.LMK[6] = data0[i * 10 + 3];
    temp_LMK.LMK[7] = data0[i * 10 + 8];
    temp_LMK.LMK[8] = data0[i * 10 + 4];
    temp_LMK.LMK[9] = data0[i * 10 + 9];
    LMK[i] = temp_LMK;
  }

  output_index = 1; // 0:point 1:bounding box 2: score
  mx_uint *shape1 = 0;
  mx_uint shape_len1;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape1, &shape_len1);

  size = 1;
  for (mx_uint i = 0; i < shape_len1; ++i) {
    size *= shape1[i];
  }

  std::vector<mx_float> data1(size);
  MXPredGetOutput(pred_hnd, output_index, &(data1[0]), size);

  x1.resize(DATA_size);
  y1.resize(DATA_size);
  x2.resize(DATA_size);
  y2.resize(DATA_size);

  #pragma omp parallel for
  for(unsigned int i = 0; i < DATA_size; i++){
    x1[i] = data1[i * 4];
    y1[i] = data1[i * 4 + 1];
    x2[i] = data1[i * 4 + 2];
    y2[i] = data1[i * 4 + 3];
  }

  output_index = 2; // 0:point 1:bounding box 2: score
  mx_uint *shape2 = 0;
  mx_uint shape_len2;

  // Get Output Result
  MXPredGetOutputShape(pred_hnd, output_index, &shape2, &shape_len2);

  size = 1;
  for (mx_uint i = 0; i < shape_len2; ++i) \
    size *= shape2[i];

  std::vector<mx_float> data2(size);
  MXPredGetOutput(pred_hnd, output_index, &(data2[0]), size);

  // Save Output Data
  for(unsigned int i = 1; i < size ; i+=2){
    score.push_back(mx_float(data2[i]));
  }
  // Release Predictor
  MXPredFree(pred_hnd);

  return 0;
}

int detection_MTCNN(Mat img, string model_dir, vector<RefineBox>& Final_Refine_Box, vector<LMK_Point>& Final_LMK){
  int MIN_DET_SIZE = 12;
  dev_type = 2;
  dev_id =0;
  //compute all scales to detect
  int H = img.rows, W = img.cols;
  double minl = min(H, W);
  vector<double> scales;
  scales.push_back(0.6);
  double m = MIN_DET_SIZE / 20.;
  minl = minl * m;
  double factor_count = 0;

  while(minl > MIN_DET_SIZE){
    scales.push_back(m * pow(0.709, factor_count));
    minl = minl * 0.709;
    factor_count = factor_count + 1;
  }


  //first stage: PNet
  // Models path for your model, you have to modify it
  std::string PNet_json_file = model_dir + "/det1-symbol.json";
  std::string PNet_param_file = model_dir + "/det1-0001.params";

  //buffer file not need to read from file each time
  BufferFile PNet_json_data(PNet_json_file);
  BufferFile PNet_param_data(PNet_param_file);

  if (PNet_json_data.GetLength() == 0 || PNet_param_data.GetLength() == 0) {
    std::cout << "PNet load fail" << std::endl;
    return -1;
  }

  vector<BoundingBox> Bounding_Box;
  #pragma omp parallel for
  for(unsigned int i = 0; i < scales.size(); i++)
    PNet_detector(img, scales[i], H, W, PNet_json_data, PNet_param_data, Bounding_Box);

  if(Bounding_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  vector<RefineBox> PNet_nms_Box(Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Bounding_Box.size(); i++){
    PNet_nms_Box[i].x1 = Bounding_Box[i].Reg_nms_x1;
    PNet_nms_Box[i].y1 = Bounding_Box[i].Reg_nms_y1;
    PNet_nms_Box[i].x2 = Bounding_Box[i].Reg_nms_x2;
    PNet_nms_Box[i].y2 = Bounding_Box[i].Reg_nms_y2;
    PNet_nms_Box[i].score = Bounding_Box[i].score;
    PNet_nms_Box[i].area = Bounding_Box[i].area;
  }

  vector<int> Pick_Bounding_Box;
  Pick_Bounding_Box = nms(PNet_nms_Box, 0.3);

  vector<BoundingBox> First_Final_Bounding_Box(Pick_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++)
    First_Final_Bounding_Box[i] = Bounding_Box[Pick_Bounding_Box[i]];

  vector<RefineBox> Refine_Box(First_Final_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < First_Final_Bounding_Box.size(); i++){
    RefineBox temp_RefineBox;
    mx_float BBw = First_Final_Bounding_Box[i].Reg_nms_x2 - First_Final_Bounding_Box[i].Reg_nms_x1 + 1;
    mx_float BBh = First_Final_Bounding_Box[i].Reg_nms_y2 - First_Final_Bounding_Box[i].Reg_nms_y1 + 1;
    temp_RefineBox.x1 = First_Final_Bounding_Box[i].Reg_nms_x1 + First_Final_Bounding_Box[i].x1 * BBw;
    temp_RefineBox.y1 = First_Final_Bounding_Box[i].Reg_nms_y1 + First_Final_Bounding_Box[i].y1 * BBh;
    temp_RefineBox.x2 = First_Final_Bounding_Box[i].Reg_nms_x2 + First_Final_Bounding_Box[i].x2 * BBw;
    temp_RefineBox.y2 = First_Final_Bounding_Box[i].Reg_nms_y2 + First_Final_Bounding_Box[i].y2 * BBh;
    temp_RefineBox.score = First_Final_Bounding_Box[i].score;
    temp_RefineBox.area = 0;
    Refine_Box[i] = temp_RefineBox;
  }

  Refine_Box = convert_to_square(Refine_Box);

  //second stage: RNet
  vector<cv::Mat> IMGDATA(Refine_Box.size());
  mx_float x1, y1, x2, y2;
  #pragma omp parallel for
  for(unsigned int i = 0; i < Refine_Box.size(); i++){
    x1 = (Refine_Box[i].x1 < 0) ? 0 : Refine_Box[i].x1;
    y1 = (Refine_Box[i].y1 < 0) ? 0 : Refine_Box[i].y1;
    x2 = (Refine_Box[i].x2 > W - 1) ? W - 1 : Refine_Box[i].x2;
    y2 = (Refine_Box[i].y2 > H - 1) ? H - 1 : Refine_Box[i].y2;

    cv::Rect r(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    cv::Mat IMG_tmp = img(r).clone();
    IMGDATA[i] = IMG_tmp;
  }

  vector<mx_float> RNet_score, RNet_x1, RNet_y1, RNet_x2, RNet_y2;

  // Models path for your model, you have to modify it
  std::string RNet_json_file = model_dir + "/det2-symbol.json";
  std::string RNet_param_file = model_dir + "/det2-0001.params";

  BufferFile RNet_json_data(RNet_json_file);
  BufferFile RNet_param_data(RNet_param_file);

  if (RNet_json_data.GetLength() == 0 || RNet_param_data.GetLength() == 0) {
    std::cout << "PNet load fail" << std::endl;
    return -1;
  }

  RNet_detector(IMGDATA, RNet_json_data, RNet_param_data, RNet_score, RNet_x1, RNet_y1, RNet_x2, RNet_y2);

  vector<RefineBox> RNet_Refine_Box;
  mx_float RNet_threshold = 0.7;
  for(unsigned int i = 0; i < Refine_Box.size(); i++){
    if(RNet_score[i] > RNet_threshold){
      Refine_Box[i].score = RNet_score[i];
      RNet_Refine_Box.push_back(Refine_Box[i]);
    }
  }

  if(RNet_Refine_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  for(unsigned int i = RNet_score.size() - 1; i >= 0; i--){
    if(RNet_score[i] <= RNet_threshold){
      RNet_x1.erase(RNet_x1.begin() + i);
      RNet_y1.erase(RNet_y1.begin() + i);
      RNet_x2.erase(RNet_x2.begin() + i);
      RNet_y2.erase(RNet_y2.begin() + i);
    }
    if(i == 0)
      break;
  }


  Pick_Bounding_Box = nms(RNet_Refine_Box, 0.4);

  vector<RefineBox> Second_Final_Refine_Box(Pick_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++){
    RefineBox temp_Refine_Box = RNet_Refine_Box[Pick_Bounding_Box[i]];
    mx_float temp_w = temp_Refine_Box.x2 - temp_Refine_Box.x1 + 1;
    mx_float temp_h = temp_Refine_Box.y2 - temp_Refine_Box.y1 + 1;

    mx_float temp_Reg_x1 = RNet_x1[Pick_Bounding_Box[i]] * temp_w;
    mx_float temp_Reg_y1 = RNet_y1[Pick_Bounding_Box[i]] * temp_h;
    mx_float temp_Reg_x2 = RNet_x2[Pick_Bounding_Box[i]] * temp_w;
    mx_float temp_Reg_y2 = RNet_y2[Pick_Bounding_Box[i]] * temp_h;

    temp_Refine_Box.x1 = temp_Refine_Box.x1 + temp_Reg_x1;
    temp_Refine_Box.y1 = temp_Refine_Box.y1 + temp_Reg_y1;
    temp_Refine_Box.x2 = temp_Refine_Box.x2 + temp_Reg_x2;
    temp_Refine_Box.y2 = temp_Refine_Box.y2 + temp_Reg_y2;

    Second_Final_Refine_Box[i] = temp_Refine_Box;
  }

  Second_Final_Refine_Box = convert_to_square(Second_Final_Refine_Box);

  //third stage: ONet
  vector<cv::Mat> ONet_IMGDATA(Second_Final_Refine_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Second_Final_Refine_Box.size(); i++){
    x1 = (Second_Final_Refine_Box[i].x1 < 0) ? 0 : Second_Final_Refine_Box[i].x1;
    y1 = (Second_Final_Refine_Box[i].y1 < 0) ? 0 : Second_Final_Refine_Box[i].y1;
    x2 = (Second_Final_Refine_Box[i].x2 > W - 1) ? W - 1 : Second_Final_Refine_Box[i].x2;
    y2 = (Second_Final_Refine_Box[i].y2 > H - 1) ? H - 1 : Second_Final_Refine_Box[i].y2;

    cv::Rect r(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    cv::Mat IMG_tmp = img(r).clone();
    ONet_IMGDATA[i] = IMG_tmp;
  }

  vector<mx_float> ONet_score, ONet_x1, ONet_y1, ONet_x2, ONet_y2;
  vector<LMK_Point> ONet_LMK;

  // Models path for your model, you have to modify it
  std::string ONet_json_file = model_dir + "/det3-symbol.json";
  std::string ONet_param_file = model_dir + "/det3-0001.params";

  BufferFile ONet_json_data(ONet_json_file);
  BufferFile ONet_param_data(ONet_param_file);

  if (ONet_json_data.GetLength() == 0 || ONet_param_data.GetLength() == 0) {
    std::cout << "PNet load fail" << std::endl;
    return -1;
  }

  ONet_detector(ONet_IMGDATA, ONet_json_data, ONet_param_data, ONet_score, ONet_x1, ONet_y1, ONet_x2, ONet_y2, ONet_LMK);

  vector<RefineBox> ONet_Refine_Box;
  mx_float ONet_threshold = 0.8;
  for(unsigned int i = 0; i < Second_Final_Refine_Box.size(); i++){
    if(ONet_score[i] > ONet_threshold){
      Second_Final_Refine_Box[i].score = ONet_score[i];
      ONet_Refine_Box.push_back(Second_Final_Refine_Box[i]);
    }
  }

  if(ONet_Refine_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  for(unsigned int i = ONet_score.size() - 1; i >= 0; i--){
    if(ONet_score[i] <= ONet_threshold){
      ONet_x1.erase(ONet_x1.begin() + i);
      ONet_y1.erase(ONet_y1.begin() + i);
      ONet_x2.erase(ONet_x2.begin() + i);
      ONet_y2.erase(ONet_y2.begin() + i);
      ONet_LMK.erase(ONet_LMK.begin() + i);
    }
    if(i == 0)
      break;
  }

  #pragma omp parallel for
  for(unsigned int i = 0; i < ONet_Refine_Box.size(); i++){
    mx_float temp_LMK_w = ONet_Refine_Box[i].x2 - ONet_Refine_Box[i].x1 + 1;
    mx_float temp_LMK_h = ONet_Refine_Box[i].y2 - ONet_Refine_Box[i].y1 + 1;

    ONet_LMK[i].LMK[0] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[0];
    ONet_LMK[i].LMK[1] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[1];
    ONet_LMK[i].LMK[2] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[2];
    ONet_LMK[i].LMK[3] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[3];
    ONet_LMK[i].LMK[4] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[4];
    ONet_LMK[i].LMK[5] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[5];
    ONet_LMK[i].LMK[6] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[6];
    ONet_LMK[i].LMK[7] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[7];
    ONet_LMK[i].LMK[8] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[8];
    ONet_LMK[i].LMK[9] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[9];

    mx_float temp_w = ONet_Refine_Box[i].x2 - ONet_Refine_Box[i].x1 + 1;
    mx_float temp_h = ONet_Refine_Box[i].y2 - ONet_Refine_Box[i].y1 + 1;

    mx_float temp_Reg_x1 = ONet_x1[i] * temp_w;
    mx_float temp_Reg_y1 = ONet_y1[i] * temp_h;
    mx_float temp_Reg_x2 = ONet_x2[i] * temp_w;
    mx_float temp_Reg_y2 = ONet_y2[i] * temp_h;

    ONet_Refine_Box[i].x1 = ONet_Refine_Box[i].x1 + temp_Reg_x1;
    ONet_Refine_Box[i].y1 = ONet_Refine_Box[i].y1 + temp_Reg_y1;
    ONet_Refine_Box[i].x2 = ONet_Refine_Box[i].x2 + temp_Reg_x2;
    ONet_Refine_Box[i].y2 = ONet_Refine_Box[i].y2 + temp_Reg_y2;
  }

  Pick_Bounding_Box = nms(ONet_Refine_Box, 0.4, 1);

  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++){
    Final_Refine_Box.push_back(ONet_Refine_Box[Pick_Bounding_Box[i]]);
    Final_LMK.push_back(ONet_LMK[Pick_Bounding_Box[i]]);
  }

  return 0;
}

cv::Mat MTCNN_Dlib_Detection(Mat IMG, string model_path, dlib::shape_predictor sp, vector<cv::Mat>& Face, vector<cv::Rect>& Bounding_Box, vector<double*>& LMK, int input_dev_type = 2, int input_dev_id = 1){
	dev_type = input_dev_type;
  dev_id = input_dev_id;

	int r_[4];
  cv::Rect expand_r;
  double mtcnn_ROIpoints[10], dlib_ROIpoints[10];
  Load_model_time = 0;

  dlib::array2d<dlib::rgb_pixel> imgD;


  //read image
  cv::Mat InpuImage = IMG.clone(), Show_IMG = IMG.clone();
  if(IMG.channels() == 1)
    cv::cvtColor(IMG, InpuImage, CV_GRAY2BGR);
  else 
    InpuImage = IMG.clone();
  Show_IMG = InpuImage.clone();
  if(!InpuImage.data){
    cout<<"-Fail to open\n";
    return cv::Mat();
  }

  dlib::assign_image(imgD, dlib::cv_image<dlib::bgr_pixel>(InpuImage));

  vector<RefineBox> Refine_Box;
  vector<LMK_Point> MTCNN_LMK;

  detection_MTCNN(InpuImage, model_path, Refine_Box, MTCNN_LMK);

  for(unsigned int i = 0; i < Refine_Box.size(); i++)
  {
    double *lmk;
    lmk = new double[136];

    r_[0] = Refine_Box[i].x1;
    r_[1] = Refine_Box[i].y1;
    r_[2] = Refine_Box[i].x2;
    r_[3] = Refine_Box[i].y2;
    cv::Rect r(r_[0],r_[1],r_[2]-r_[0],r_[3]-r_[1]);

    cv::rectangle(Show_IMG, r, cv::Scalar(255, 255, 255));
    dlib::matrix<float,0,1> DlibinniSp = sp.getInitShape(dlib_ROIpoints);
    LandmarksToROI_(mtcnn_ROIpoints,dlib_ROIpoints,MTCNN_LMK[i].LMK,r);
    warpAffineLandmarks(DlibinniSp,getLandmarkAffineMatrix(dlib_ROIpoints, mtcnn_ROIpoints),r.width,r.height);
    dlib::full_object_detection shape = sp(imgD, dlib::rectangle(r_[0],r_[1],r_[2],r_[3]), DlibinniSp);

    for(unsigned int j = 0 ; j < 68 ; j++){
      lmk[2*j] = (double)(shape.part(j).x());
      lmk[2*j+1] = (double)(shape.part(j).y());
    }

    Bounding_Box.push_back(r);
    LMK.push_back(lmk);
    Face.push_back(InpuImage(r).clone());
  }
  return Show_IMG;
}

cv::Mat MTCNN_Dlib_Detection(string img_path, string model_path, dlib::shape_predictor sp, vector<cv::Mat>& Face, vector<cv::Rect>& Bounding_Box, vector<double*>& LMK, int input_dev_type = 2, int input_dev_id = 1){
	Mat InpuImage = cv::imread(img_path.c_str());
	Mat Show_IMG = MTCNN_Dlib_Detection(InpuImage, model_path, sp, Face, Bounding_Box, LMK, input_dev_type, input_dev_id);
	return Show_IMG;
}


/* -------------------------------------------------------------------- */
// 2017/11/30 by Lynn
int detection_MTCNN(Mat& img, vector<RefineBox>& Final_Refine_Box, vector<LMK_Point>& Final_LMK, \
  PreLoadPNetPool& toFasterPNet, \
  BufferFile& RNet_json_data, BufferFile& RNet_param_data, \
  BufferFile& ONet_json_data, BufferFile& ONet_param_data)
{
  // int MIN_DET_SIZE = 12;
  dev_type = 2;
  dev_id =0;
  //compute all scales to detect
  int H = img.rows, W = img.cols;
  // double minl = min(H, W);
  // vector<double> scales;
  // scales.push_back(0.6);
  // double m = MIN_DET_SIZE / 20.;
  // minl = minl * m;
  // double factor_count = 0;

  // while(minl > MIN_DET_SIZE){
  //   scales.push_back(m * pow(0.709, factor_count));
  //   minl = minl * 0.709;
  //   factor_count = factor_count + 1;
  // }


  //==========================================+
  //                                          |
  //            First stage: PNet             |
  //                                          |
  //==========================================+
  // Models path for your model, you have to modify it
  // std::string PNet_json_file = model_dir + "/det1-symbol.json";
  // std::string PNet_param_file = model_dir + "/det1-0001.params";

  // BufferFile PNet_json_data(PNet_json_file);
  // BufferFile PNet_param_data(PNet_param_file);
  // if (PNet_json_data.GetLength() == 0 || PNet_param_data.GetLength() == 0) {
  //   std::cout << "PNet load fail" << std::endl;
  //   return -1;
  // }


  vector<BoundingBox> Bounding_Box;
  #pragma omp parallel for
  for(unsigned int i = 0; i < toFasterPNet.scales.size(); i++)
  {
    PNet_detector(img, toFasterPNet.scales[i], H, W, toFasterPNet.md_PNet[i], Bounding_Box);
    if(toShowTimeLog == 1)cout<<"\ttime - PNet_detector(), scale = "<<toFasterPNet.scales[i]<<":\t"<<TimeGoesBy()<<endl;
  }

  // if(toShowTimeLog == 1)cout<<"\ttime - PNet_detector():\t"<<TimeGoesBy()<<endl;
  
  if(Bounding_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  vector<RefineBox> PNet_nms_Box(Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Bounding_Box.size(); i++){
    PNet_nms_Box[i].x1 = Bounding_Box[i].Reg_nms_x1;
    PNet_nms_Box[i].y1 = Bounding_Box[i].Reg_nms_y1;
    PNet_nms_Box[i].x2 = Bounding_Box[i].Reg_nms_x2;
    PNet_nms_Box[i].y2 = Bounding_Box[i].Reg_nms_y2;
    PNet_nms_Box[i].score = Bounding_Box[i].score;
    PNet_nms_Box[i].area = Bounding_Box[i].area;
  }

  vector<int> Pick_Bounding_Box;
  Pick_Bounding_Box = nms(PNet_nms_Box, 0.3);

  vector<BoundingBox> First_Final_Bounding_Box(Pick_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++)
    First_Final_Bounding_Box[i] = Bounding_Box[Pick_Bounding_Box[i]];

  vector<RefineBox> Refine_Box(First_Final_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < First_Final_Bounding_Box.size(); i++){
    RefineBox temp_RefineBox;
    mx_float BBw = First_Final_Bounding_Box[i].Reg_nms_x2 - First_Final_Bounding_Box[i].Reg_nms_x1 + 1;
    mx_float BBh = First_Final_Bounding_Box[i].Reg_nms_y2 - First_Final_Bounding_Box[i].Reg_nms_y1 + 1;
    temp_RefineBox.x1 = First_Final_Bounding_Box[i].Reg_nms_x1 + First_Final_Bounding_Box[i].x1 * BBw;
    temp_RefineBox.y1 = First_Final_Bounding_Box[i].Reg_nms_y1 + First_Final_Bounding_Box[i].y1 * BBh;
    temp_RefineBox.x2 = First_Final_Bounding_Box[i].Reg_nms_x2 + First_Final_Bounding_Box[i].x2 * BBw;
    temp_RefineBox.y2 = First_Final_Bounding_Box[i].Reg_nms_y2 + First_Final_Bounding_Box[i].y2 * BBh;
    temp_RefineBox.score = First_Final_Bounding_Box[i].score;
    temp_RefineBox.area = 0;
    Refine_Box[i] = temp_RefineBox;
  }

  Refine_Box = convert_to_square(Refine_Box);

  if(toShowTimeLog == 1)cout<<"\ttime - get PNet results:\t"<<TimeGoesBy()<<endl;

  //==========================================+
  //                                          |
  //           Second stage: RNet             |
  //                                          |
  //==========================================+
  vector<cv::Mat> IMGDATA(Refine_Box.size());
  mx_float x1, y1, x2, y2;
  // #pragma omp parallel for
  for(unsigned int i = 0; i < Refine_Box.size(); i++){
    x1 = (Refine_Box[i].x1 < 0) ? 0 : Refine_Box[i].x1;
    y1 = (Refine_Box[i].y1 < 0) ? 0 : Refine_Box[i].y1;
    x2 = (Refine_Box[i].x2 > W - 1) ? W - 1 : Refine_Box[i].x2;
    y2 = (Refine_Box[i].y2 > H - 1) ? H - 1 : Refine_Box[i].y2;

    cv::Rect r(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
    if (!( 0 <= r.x && 0 <= r.width && r.x + r.width <= img.cols && 0 <= r.y && 0 <= r.height && r.y + r.height <= img.rows))
    {
      Refine_Box.erase(Refine_Box.begin()+i);
      i--;
    }
    else
    {
      cv::Mat IMG_tmp = img(r).clone();
      IMGDATA[i] = IMG_tmp;
    }
  }
  IMGDATA.resize(Refine_Box.size());

  vector<mx_float> RNet_score, RNet_x1, RNet_y1, RNet_x2, RNet_y2;

  // Models path for your model, you have to modify it
  // std::string RNet_json_file = model_dir + "/det2-symbol.json";
  // std::string RNet_param_file = model_dir + "/det2-0001.params";

  // BufferFile RNet_json_data(RNet_json_file);
  // BufferFile RNet_param_data(RNet_param_file);

  // if (RNet_json_data.GetLength() == 0 || RNet_param_data.GetLength() == 0) {
  //   std::cout << "RNet load fail" << std::endl;
  //   return -1;
  // }

  RNet_detector(IMGDATA, RNet_json_data, RNet_param_data, RNet_score, RNet_x1, RNet_y1, RNet_x2, RNet_y2);

  if(toShowTimeLog == 1)cout<<"\tSize - RNet input:\t"<<IMGDATA.size()<<endl;
  if(toShowTimeLog == 1)cout<<"\ttime - RNet_detector():\t"<<TimeGoesBy()<<endl;

  vector<RefineBox> RNet_Refine_Box;
  mx_float RNet_threshold = 0.7;
  for(unsigned int i = 0; i < Refine_Box.size(); i++){
    if(RNet_score[i] > RNet_threshold){
      Refine_Box[i].score = RNet_score[i];
      RNet_Refine_Box.push_back(Refine_Box[i]);
    }
  }

  if(RNet_Refine_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  for(unsigned int i = RNet_score.size() - 1; i >= 0; i--){
    if(RNet_score[i] <= RNet_threshold){
      RNet_x1.erase(RNet_x1.begin() + i);
      RNet_y1.erase(RNet_y1.begin() + i);
      RNet_x2.erase(RNet_x2.begin() + i);
      RNet_y2.erase(RNet_y2.begin() + i);
    }
    if(i == 0)
      break;
  }


  Pick_Bounding_Box = nms(RNet_Refine_Box, 0.4);

  vector<RefineBox> Second_Final_Refine_Box(Pick_Bounding_Box.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++){
    RefineBox temp_Refine_Box = RNet_Refine_Box[Pick_Bounding_Box[i]];
    mx_float temp_w = temp_Refine_Box.x2 - temp_Refine_Box.x1 + 1;
    mx_float temp_h = temp_Refine_Box.y2 - temp_Refine_Box.y1 + 1;

    mx_float temp_Reg_x1 = RNet_x1[Pick_Bounding_Box[i]] * temp_w;
    mx_float temp_Reg_y1 = RNet_y1[Pick_Bounding_Box[i]] * temp_h;
    mx_float temp_Reg_x2 = RNet_x2[Pick_Bounding_Box[i]] * temp_w;
    mx_float temp_Reg_y2 = RNet_y2[Pick_Bounding_Box[i]] * temp_h;

    temp_Refine_Box.x1 = temp_Refine_Box.x1 + temp_Reg_x1;
    temp_Refine_Box.y1 = temp_Refine_Box.y1 + temp_Reg_y1;
    temp_Refine_Box.x2 = temp_Refine_Box.x2 + temp_Reg_x2;
    temp_Refine_Box.y2 = temp_Refine_Box.y2 + temp_Reg_y2;

    Second_Final_Refine_Box[i] = temp_Refine_Box;
  }

  Second_Final_Refine_Box = convert_to_square(Second_Final_Refine_Box);

  //==========================================+
  //                                          |
  //            Third stage: ONet             |
  //                                          |
  //==========================================+
  vector<cv::Mat> ONet_IMGDATA(Second_Final_Refine_Box.size());
  // #pragma omp parallel for
  for(unsigned int i = 0; i < Second_Final_Refine_Box.size(); i++){
    x1 = (Second_Final_Refine_Box[i].x1 < 0) ? 0 : Second_Final_Refine_Box[i].x1;
    y1 = (Second_Final_Refine_Box[i].y1 < 0) ? 0 : Second_Final_Refine_Box[i].y1;
    x2 = (Second_Final_Refine_Box[i].x2 > W - 1) ? W - 1 : Second_Final_Refine_Box[i].x2;
    y2 = (Second_Final_Refine_Box[i].y2 > H - 1) ? H - 1 : Second_Final_Refine_Box[i].y2;

    cv::Rect r(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    if (!( 0 <= r.x && 0 <= r.width && r.x + r.width <= img.cols && 0 <= r.y && 0 <= r.height && r.y + r.height <= img.rows))
    {
      Second_Final_Refine_Box.erase(Second_Final_Refine_Box.begin()+i);
      i--;
    }
    else
    {
      cv::Mat IMG_tmp = img(r).clone();
      ONet_IMGDATA[i] = IMG_tmp;
    }
  }
  //printf("core dump~~~ why 2\n");


  ONet_IMGDATA.resize(Second_Final_Refine_Box.size());

  if(Second_Final_Refine_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }
  

  vector<mx_float> ONet_score, ONet_x1, ONet_y1, ONet_x2, ONet_y2;
  vector<LMK_Point> ONet_LMK;

  if(toShowTimeLog == 1)cout<<"\ttime - get RNet results:\t"<<TimeGoesBy()<<endl;

  // Models path for your model, you have to modify it
  // std::string ONet_json_file = model_dir + "/det3-symbol.json";
  // std::string ONet_param_file = model_dir + "/det3-0001.params";

  // BufferFile ONet_json_data(ONet_json_file);
  // BufferFile ONet_param_data(ONet_param_file);

  // if (ONet_json_data.GetLength() == 0 || ONet_param_data.GetLength() == 0) {
  //   std::cout << "ONet load fail" << std::endl;
  //   return -1;
  // }

  //printf("core dump~~~ why 3\n");

  ONet_detector(ONet_IMGDATA, ONet_json_data, ONet_param_data, ONet_score, ONet_x1, ONet_y1, ONet_x2, ONet_y2, ONet_LMK);

  //printf("core dump~~~ why 4\n");

  if(toShowTimeLog == 1)cout<<"\tSize - ONet input:\t"<<ONet_IMGDATA.size()<<endl;
  if(toShowTimeLog == 1)cout<<"\ttime - ONet_detector():\t"<<TimeGoesBy()<<endl;

  vector<RefineBox> ONet_Refine_Box;
  mx_float ONet_threshold = 0.8;
  for(unsigned int i = 0; i < Second_Final_Refine_Box.size(); i++){
    if(ONet_score[i] > ONet_threshold){
      Second_Final_Refine_Box[i].score = ONet_score[i];
      ONet_Refine_Box.push_back(Second_Final_Refine_Box[i]);
    }
  }

  //printf("core dump~~~ why 5\n");

  if(ONet_Refine_Box.size() == 0){
    //cout << "Can not find any face." << endl;
    return -1;
  }

  for(unsigned int i = ONet_score.size() - 1; i >= 0; i--){
    if(ONet_score[i] <= ONet_threshold){
      ONet_x1.erase(ONet_x1.begin() + i);
      ONet_y1.erase(ONet_y1.begin() + i);
      ONet_x2.erase(ONet_x2.begin() + i);
      ONet_y2.erase(ONet_y2.begin() + i);
      ONet_LMK.erase(ONet_LMK.begin() + i);
    }
    if(i == 0)
      break;
  }

  //printf("core dump~~~ why 6\n");


  #pragma omp parallel for
  for(unsigned int i = 0; i < ONet_Refine_Box.size(); i++){
    mx_float temp_LMK_w = ONet_Refine_Box[i].x2 - ONet_Refine_Box[i].x1 + 1;
    mx_float temp_LMK_h = ONet_Refine_Box[i].y2 - ONet_Refine_Box[i].y1 + 1;

    ONet_LMK[i].LMK[0] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[0];
    ONet_LMK[i].LMK[1] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[1];
    ONet_LMK[i].LMK[2] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[2];
    ONet_LMK[i].LMK[3] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[3];
    ONet_LMK[i].LMK[4] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[4];
    ONet_LMK[i].LMK[5] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[5];
    ONet_LMK[i].LMK[6] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[6];
    ONet_LMK[i].LMK[7] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[7];
    ONet_LMK[i].LMK[8] = ONet_Refine_Box[i].x1 + temp_LMK_w * ONet_LMK[i].LMK[8];
    ONet_LMK[i].LMK[9] = ONet_Refine_Box[i].y1 + temp_LMK_h * ONet_LMK[i].LMK[9];

    mx_float temp_w = ONet_Refine_Box[i].x2 - ONet_Refine_Box[i].x1 + 1;
    mx_float temp_h = ONet_Refine_Box[i].y2 - ONet_Refine_Box[i].y1 + 1;

    mx_float temp_Reg_x1 = ONet_x1[i] * temp_w;
    mx_float temp_Reg_y1 = ONet_y1[i] * temp_h;
    mx_float temp_Reg_x2 = ONet_x2[i] * temp_w;
    mx_float temp_Reg_y2 = ONet_y2[i] * temp_h;

    ONet_Refine_Box[i].x1 = ONet_Refine_Box[i].x1 + temp_Reg_x1;
    ONet_Refine_Box[i].y1 = ONet_Refine_Box[i].y1 + temp_Reg_y1;
    ONet_Refine_Box[i].x2 = ONet_Refine_Box[i].x2 + temp_Reg_x2;
    ONet_Refine_Box[i].y2 = ONet_Refine_Box[i].y2 + temp_Reg_y2;
  }

  //printf("core dump~~~ why 7\n");

  Pick_Bounding_Box = nms(ONet_Refine_Box, 0.4, 1);

  for(unsigned int i = 0; i < Pick_Bounding_Box.size(); i++){
    Final_Refine_Box.push_back(ONet_Refine_Box[Pick_Bounding_Box[i]]);
    Final_LMK.push_back(ONet_LMK[Pick_Bounding_Box[i]]);
  }

  //printf("core dump~~~ why 8\n");

  if(toShowTimeLog == 1)cout<<"\ttime - get ONet results:\t"<<TimeGoesBy()<<endl;

  return 0;
}

/* --------------------------------------------------------------------- */
// 2017/10/03 by Lynn
int MTCNN_Dlib_Detection(Mat& IMG, string model_path, dlib::shape_predictor& sp, vector<cv::Rect>& Bounding_Box, vector<double*>& LMK, \
  PreLoadPNetPool& toFasterPNet, \
  BufferFile& RNet_json_data, BufferFile& RNet_param_data, \
  BufferFile& ONet_json_data, BufferFile& ONet_param_data, \
  int input_dev_type = 2, int input_dev_id = 1)
{

  Bounding_Box.clear();
  LMK.clear();

  dev_type = input_dev_type;
  dev_id = input_dev_id;

  int r_[4];
  cv::Rect expand_r;
  double mtcnn_ROIpoints[10], dlib_ROIpoints[10];
  Load_model_time = 0;

  dlib::array2d<dlib::rgb_pixel> imgD;

  //read image
  cv::Mat InpuImage;
  if(IMG.channels() == 1) cv::cvtColor(IMG, InpuImage, CV_GRAY2BGR);
  else InpuImage = IMG.clone();
  dlib::assign_image(imgD, dlib::cv_image<dlib::bgr_pixel>(InpuImage));

  vector<RefineBox> Refine_Box;
  vector<LMK_Point> MTCNN_LMK;

  detection_MTCNN(InpuImage, Refine_Box, MTCNN_LMK,toFasterPNet,RNet_json_data,RNet_param_data,ONet_json_data,ONet_param_data);

// cout<<"Detection time:\tdetection_MTCNN()\t"<<TimeGoesBy()<<endl;

  int face_numbers = Refine_Box.size();
  for(unsigned int i = 0; i < face_numbers; i++)
  {
    double *lmk;
    lmk = new double[136];

    r_[0] = Refine_Box[i].x1;
    r_[1] = Refine_Box[i].y1;
    r_[2] = Refine_Box[i].x2;
    r_[3] = Refine_Box[i].y2;
    cv::Rect r(r_[0],r_[1],r_[2]-r_[0],r_[3]-r_[1]);
    
    
    // // use for Lynn thesis
    // char thssname[755];
    // cv::Mat img_mtcnn_result = IMG.clone();
    // cv::Mat img_ini_dlib = IMG.clone();
    // cv::Mat img_affine_dlib = IMG.clone();
    // cv::Mat img_MTCNN_dlib = IMG.clone();
    // for(int ki = 0;ki<5;ki++)
    //   cv::circle(img_mtcnn_result, cv::Point(MTCNN_LMK[i].LMK[ki*2], MTCNN_LMK[i].LMK[ki*2+1]), 3, cv::Scalar(255,0,0), -1);
    // sprintf(thssname, "D:/Project/z_LynnThesis_use/04 DrawMTCNN_Dlib tf/result/%d_02_MTCNN_Point.png",i);
    // cv::imwrite(thssname,img_mtcnn_result);
    
    // dlib::matrix<float,0,1> DlibinniSp_forThesis = sp.getInitShape(dlib_ROIpoints);
    // for(int ki = 0;ki<68;ki++)
    //   cv::circle(img_ini_dlib, cv::Point(DlibinniSp_forThesis(ki*2)*r.width+r.x, DlibinniSp_forThesis(ki*2+1)*r.height+r.y), 3, cv::Scalar(0,255,0), -1);
    // sprintf(thssname, "D:/Project/z_LynnThesis_use/04 DrawMTCNN_Dlib tf/result/%d_01_Dlib_init.png",i);
    // cv::imwrite(thssname,img_ini_dlib);
    // // use for Lynn thesis


    dlib::matrix<float,0,1> DlibinniSp = sp.getInitShape(dlib_ROIpoints);
    LandmarksToROI_(mtcnn_ROIpoints,dlib_ROIpoints,MTCNN_LMK[i].LMK,r);
    warpAffineLandmarks(DlibinniSp,getLandmarkAffineMatrix(dlib_ROIpoints, mtcnn_ROIpoints),r.width,r.height);

    // // use for Lynn thesis
    // for(int ki = 0;ki<68;ki++)
    //   cv::circle(img_affine_dlib, cv::Point(DlibinniSp(ki*2)*r.width+r.x, DlibinniSp(ki*2+1)*r.height+r.y), 3, cv::Scalar(255,180,0), -1);
    // sprintf(thssname, "D:/Project/z_LynnThesis_use/04 DrawMTCNN_Dlib tf/result/%d_03_Dlib_affineTF.png",i);
    // cv::imwrite(thssname,img_affine_dlib);
    // // use for Lynn thesis


    dlib::full_object_detection shape = sp(imgD, dlib::rectangle(r_[0],r_[1],r_[2],r_[3]), DlibinniSp);

    for(unsigned int j = 0 ; j < 68 ; j++){
      lmk[2*j] = (double)(shape.part(j).x());
      lmk[2*j+1] = (double)(shape.part(j).y());
    }
    LMK.push_back(lmk);
    Bounding_Box.push_back(r);


    // // use for Lynn thesis
    // for(int ki = 0;ki<68;ki++)
    //   cv::circle(img_MTCNN_dlib, cv::Point(lmk[ki*2], lmk[ki*2+1]), 3, cv::Scalar(0,0,255), -1);
    // sprintf(thssname, "D:/Project/z_LynnThesis_use/04 DrawMTCNN_Dlib tf/result/%d_04_MTCNNDlib_pts.png",i);
    // cv::imwrite(thssname,img_MTCNN_dlib);
    // // use for Lynn thesis
  }
  return face_numbers;
}
