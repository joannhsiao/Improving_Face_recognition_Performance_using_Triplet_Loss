#pragma once
#include <opencv2/core/core.hpp>
#include <string>

typedef struct {
  int CameraID;
  int Width;
  int Height;
} CameraSettings;

typedef enum {
  IdentificationState, EnterNameState, DowncountState, RegImageAcqState, SelRegImageState, InputNameState, RegistrationState, RegistFeatureState,
} SystemState;

extern SystemState state;
extern bool bLButtonDown, bGrid, bSelPerson;
extern cv::Point mousePnt;
extern std::string CanvasName;
extern cv::Rect SelPerson;

extern void StreamLoop(CameraSettings* camera, LFQueue1P1C<cv::Mat> *imageQueue, bool *shutdown);
extern double get_wall_time();
extern bool RegImage(LFQueue1P1C<cv::Mat>& imageQueue);