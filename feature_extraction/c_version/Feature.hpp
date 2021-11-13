#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <list>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>

#include <sqlite3.h>
#include "boost/progress.hpp"
#include "boost/timer.hpp"

#include "FileProc.hpp"
#include "FileProcDNN.hpp"

#include "psql_handler.hpp"

const int fvSize = 342;

const char face_db_data_root[50] = "/home/csieface/FaceSystem/FaceID_DataEngine/";





//------------------------ 2018/09/09 , Lynn
class Person {
public:
  int pid;
  char name[256];
  char title[256];
  char email[60];
  char student_id[50];
  char card_id[50];
  char profile_img[FILENAME_MAX];
  int pflag;
  Person():pid(-1),pflag(0),name(""),title(""),email(""),student_id(""),card_id(""),profile_img(""){}
  ~Person(){}
  Person(const Person& src)
  {
    pid = src.pid;
    pflag = src.pflag;
    std::strcpy(name, src.name);
    std::strcpy(title, src.title);
    std::strcpy(email, src.email);
    std::strcpy(student_id, src.student_id);
    std::strcpy(card_id, src.card_id);
    std::strcpy(profile_img, src.profile_img);
  }
  Person& operator=(const Person& src)
  {
    pid = src.pid;
    pflag = src.pflag;
    std::strcpy(name, src.name);
    std::strcpy(title, src.title);
    std::strcpy(email, src.email);
    std::strcpy(student_id, src.student_id);
    std::strcpy(card_id, src.card_id);
    std::strcpy(profile_img, src.profile_img);
    return *this;
  }
  bool operator==(const Person& src) const
  {
    // if(std::strcmp(name, src.name))return false;
    // if(std::strcmp(title, src.title))return false;
    // if(std::strcmp(email, src.email))return false;
    // if(std::strcmp(student_id, src.student_id))return false;
    // if(std::strcmp(card_id, src.card_id))return false;
    // if(std::strcmp(profile_img, src.profile_img))return false;
    if(!(pid == src.pid)) return false;
    return true;
  }
  bool operator!=(const Person& src) const
  {
    return !((*this) == src);
  }
  bool operator>(const Person& src) const
  {
    return (pid > src.pid);
    // return (std::strcmp(student_id, src.student_id)>0) || (pid > src.pid);
  }
  bool operator>=(const Person& src) const
  {
    return (pid >= src.pid);
    // return (std::strcmp(student_id, src.student_id)>=0) || (pid >= src.pid);
  }
  bool operator<(const Person& src) const
  {
    return (pid < src.pid);
    // return (std::strcmp(student_id, src.student_id)<0) || (pid < src.pid);
  }
  bool operator<=(const Person& src) const
  {
    return (pid <= src.pid);
    // return (std::strcmp(student_id, src.student_id)<=0) || (pid <= src.pid);
  }
};

class FeatureVec {
public:
  int fid;
  float sqrt_simdot;
  float fv[fvSize];
  char img_path[FILENAME_MAX];
  cv::Mat I;
  FeatureVec():fid(-1){}
  ~FeatureVec(){}
  FeatureVec(const FeatureVec& src)
  {
    fid = src.fid;
    sqrt_simdot = src.sqrt_simdot;
    std::memcpy(fv, src.fv, sizeof(float)*fvSize);
    std::strcpy(img_path, src.img_path);
    I = src.I.clone();
  }
  FeatureVec& operator=(const FeatureVec& src)
  {
    fid = src.fid;
    sqrt_simdot = src.sqrt_simdot;
    std::memcpy(fv, src.fv, sizeof(float)*fvSize);
    std::strcpy(img_path, src.img_path);
    I = src.I.clone();
    return *this;
  }
  bool operator==(const FeatureVec& src) const
  {
    // if(!(sqrt_simdot == src.sqrt_simdot))return false;
    // if(std::memcmp(fv, src.fv,sizeof(float)*fvSize))return false;
    if(std::strcmp(img_path, src.img_path))return false;
    if(!(fid == src.fid))return false;
    return true;
  }
  bool operator!=(const FeatureVec& src) const
  {
    return !((*this) == src);
  }
};
typedef std::vector<FeatureVec> FVecV;
typedef std::map<Person, FVecV> P_FV;
//------------------------ 2018/09/09 , Lynn





struct FeatureImage{
  float fv[fvSize+1];
  float sqrt_simdot;
  cv::Mat I;
  cv::Rect r;
  char name[255];
};

typedef std::vector<FeatureImage> FVV;


PredictorHandle Feature_Net(BufferFile& Feature_Net_json_data, BufferFile& Feature_Net_param_data, int IMG_SIZE, char *layer){
  PredictorHandle pred_hnd = 0;
  int dev_type = 2;  // 1: cpu, 2: gpu
  int dev_id = 0;  // arbitrary.
  const char* input_key[1] = {"data"};
  const char** input_keys = input_key;

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 1, static_cast<mx_uint>(1), static_cast<mx_uint>(IMG_SIZE), static_cast<mx_uint>(IMG_SIZE) };
  
  MXPredCreatePartialOut ((const char*)Feature_Net_json_data.GetBuffer(),
                          (const char*)Feature_Net_param_data.GetBuffer(),
                          static_cast<size_t>(Feature_Net_param_data.GetLength()),
                          dev_type,
                          dev_id,
                          1,
                          input_keys,
                          input_shape_indptr,
                          input_shape_data,
                          1,
                          (const char**)&layer,
                          &pred_hnd);
  assert(pred_hnd);
  return pred_hnd;
}

void Feature_Extract_exe(cv::Mat Extract_IMG, float* Feature_Vector, PredictorHandle pred_hnd){
  int IMG_Size = Extract_IMG.rows * Extract_IMG.cols;
  std::vector<mx_float> image_datas = std::vector<mx_float>(IMG_Size);
  Convert_IMG_Data(Extract_IMG, image_datas.data());
  MXPredSetInput(pred_hnd, "data", image_datas.data(), IMG_Size);
  MXPredForward(pred_hnd);
  mx_uint output_index = 0;
  mx_uint *shape = 0;
  mx_uint shape_length;
  MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_length);
  size_t size = 1;
  for (mx_uint i = 0; i < shape_length; ++i) 
    size *= shape[i];
  std::vector<float> data(size);
  MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);
  std::copy( data.begin(), data.end(), Feature_Vector);
}

int Feature_Extract(std::map <std::string, std::string> Configs, cv::Mat Extract_IMG, float* Feature_Vector){
  std::string Model_Dir = Configs["Model_Dir"];
  int Extract_IMG_Size = atoi(Configs["Extract_IMG_Size"].c_str());
  int Feature_Vector_Size = atoi(Configs["Feature_Vector_Size"].c_str());
  char* Feature_Layer = (char*)Configs["Feature_Layer"].c_str();

  BufferFile json_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".json");
  BufferFile param_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return -1;
  }

  PredictorHandle pred_hnd = Feature_Net(json_data, param_data, Extract_IMG_Size, Feature_Layer);
  Feature_Extract_exe(Extract_IMG, Feature_Vector, pred_hnd);
  MXPredFree(pred_hnd);
  return 0;
}

bool Load_Identify_model(std::map <std::string, std::string> Configs, PredictorHandle &pred_hnd)
{
  std::string Model_Name = Configs["Model_Dir"] + Configs["Model_Name"] ;
  int Extract_IMG_Size = atoi(Configs["Extract_IMG_Size"].c_str());
  char* Feature_Layer = (char*)Configs["Feature_Layer"].c_str();

  BufferFile json_data(Model_Name + ".json");
  BufferFile param_data(Model_Name + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return false;
  }
  pred_hnd = Feature_Net(json_data, param_data, Extract_IMG_Size, Feature_Layer);
  return true;
}

PredictorHandle Load_Identify_model(std::map <std::string, std::string> Configs)
{
  std::string Model_Name = Configs["Model_Dir"] + Configs["Model_Name"] ;
  int Extract_IMG_Size = atoi(Configs["Extract_IMG_Size"].c_str());
  char* Feature_Layer = (char*)Configs["Feature_Layer"].c_str();

  BufferFile json_data(Model_Name + ".json");
  BufferFile param_data(Model_Name + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return false;
  }
  return Feature_Net(json_data, param_data, Extract_IMG_Size, Feature_Layer);
}


void FeatExet(cv::Mat img2, std::string name, float *v2, PredictorHandle Network)
{
  int image_size = img2.rows * img2.cols ;
  std::vector<mx_float> image_datas = std::vector<mx_float>(image_size);
  GetImageFile(img2, name, image_datas.data());
  MXPredSetInput(Network, "data", image_datas.data(), image_size);
  MXPredForward(Network);
  mx_uint output_index = 0;
  mx_uint *shape = 0;
  mx_uint shape_len;
  MXPredGetOutputShape(Network, output_index, &shape, &shape_len);
  size_t size = 1;
  for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];
  std::vector<float> data(size);
  MXPredGetOutput(Network, output_index, &(data[0]), size);
  std::copy( data.begin(), data.end(), v2 );
}

float simd_dot(const float* x, const float* y, const long& len) {
  float inner_prod = 0.0f;
  __m128 X, Y; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
    inner_prod += x[i] * y[i];
  }
  return inner_prod;
}

int Compare_Face_From_DB(std::map <std::string, std::string> Configs, float* Feature_Vector, int Feature_Vector_Size, std::vector<std::pair<float, std::string>>& Face_Maps, int Face_Maps_Size){
  float Feature_simd_dot = sqrt(simd_dot(Feature_Vector, Feature_Vector, Feature_Vector_Size));
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  sqlite3_stmt* stmt;
  float sim , sim_th = atof(Configs["sim_th"].c_str());
  std::string Query = "SELECT * FROM  `reg_face`;";
  sqlite3_prepare_v2(Register_db, Query.c_str(), -1, &stmt, 0);

  while (sqlite3_step(stmt) == SQLITE_ROW)
  {
    char* img_path = (char*)sqlite3_column_text(stmt, 1);
    float* Feature_tmp = (float*)sqlite3_column_blob(stmt, 2);
    int test_int = (int)sqlite3_column_bytes(stmt, 2);
    float Feature_tmp_simd_dot = (float)sqlite3_column_double(stmt, 3);

    sim = NAN;
    sim = simd_dot(Feature_tmp, Feature_Vector, Feature_Vector_Size) / (Feature_tmp_simd_dot * Feature_simd_dot);
    if (std::isnan(sim)){
      std::cout << "no reg data" << std::endl;
      return -1;
    }
    if(sim < sim_th)
      continue;
    
    std::vector<std::string> IMG_Path_tmp = split_string(img_path, "/");
    std::string name = IMG_Path_tmp[IMG_Path_tmp.size() - 2] + '/' + IMG_Path_tmp[IMG_Path_tmp.size() - 1];
    if (Face_Maps.size() < Face_Maps_Size)
    {
      std::pair<float, std::string> tmp_pair(sim, name);
      Face_Maps.push_back(tmp_pair);
      sort(Face_Maps.begin(), Face_Maps.end(), strict_weak_ordering);
    }
    else
    {
      if (Face_Maps.at(0).first < sim)
      {
        Face_Maps.at(0).first = sim;
        Face_Maps.at(0).second = name;
        sort(Face_Maps.begin(), Face_Maps.end(), strict_weak_ordering);
      }
    }
  }
  sqlite3_reset(stmt);
  sqlite3_finalize(stmt);
  sqlite3_close(Register_db);
  return 0;
}

int Compare_Face_From_DB(FVV &Mbs, float sim_th, float* Feature_Vector){
  float Feature_simd_dot = sqrt(simd_dot(Feature_Vector, Feature_Vector, fvSize));
  float sim , maxsim=0;//, sim_th = atof(Configs["sim_th"].c_str());
  int all_size = Mbs.size() , rtidx = -1;
  #pragma omp parallel for
  for (int xi=0;xi<all_size;xi++)
  {
    // char* img_path = Mbs[xi].name;
    float* Feature_tmp = Mbs[xi].fv;
    float Feature_tmp_simd_dot = Mbs[xi].sqrt_simdot;

    sim = NAN;
    sim = simd_dot(Feature_tmp, Feature_Vector, fvSize) / (Feature_tmp_simd_dot * Feature_simd_dot);
    if (std::isnan(sim)){
      std::cout << "no reg data" << std::endl;
      continue;
    }
    if(sim < sim_th)
      continue;
    
    #pragma omp critical
    {
      if(sim > maxsim)
      {
        maxsim = sim;
        rtidx = xi;
      }
    }
    // std::vector<std::string> IMG_Path_tmp = split_string(img_path, "/");
    // std::string name = IMG_Path_tmp[IMG_Path_tmp.size() - 2] + '/' + IMG_Path_tmp[IMG_Path_tmp.size() - 1];
    // if (Face_Maps.size() < Face_Maps_Size)
    // {
    //   std::pair<float, std::string> tmp_pair(sim, name);
    //   Face_Maps.push_back(tmp_pair);
    //   sort(Face_Maps.begin(), Face_Maps.end(), strict_weak_ordering);
    // }
    // else
    // {
    //   if (Face_Maps.at(0).first < sim)
    //   {
    //     Face_Maps.at(0).first = sim;
    //     Face_Maps.at(0).second = name;
    //     sort(Face_Maps.begin(), Face_Maps.end(), strict_weak_ordering);
    //   }
    // }
  }
  return rtidx;
}
bool Select_Registed_Data(const char* dbname, FVV &Mbs)
{
  Mbs.clear();
  
  sqlite3* dbr;
  sqlite3_stmt* stmt;
  char q_select_members[77] = "SELECT * FROM  `reg_face`;";
  
  sqlite3_open(dbname, &dbr);
  sqlite3_prepare_v2(dbr, q_select_members, -1, &stmt, 0);
  while (sqlite3_step(stmt) == SQLITE_ROW)
  {
    int pid = (int)sqlite3_column_int(stmt, 0);
    char* pname = (char*)sqlite3_column_text(stmt, 1);
    float* tmpFVr = (float*)sqlite3_column_blob(stmt, 2);
    float Feature_tmp_simd_dot = (float)sqlite3_column_double(stmt, 3);
    
    FeatureImage tmpfv;
    std::memcpy(tmpfv.fv, tmpFVr, sizeof(float)*fvSize);
    tmpfv.sqrt_simdot = Feature_tmp_simd_dot;
    std::strcpy(tmpfv.name, pname);
    Mbs.push_back(tmpfv);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(dbr);
  return true;
}

static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
  int i;
  #pragma omp parallel
  for (i = 0; i < argc; i++) {
    printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
  }
  printf("\n");
  return 0;
}

int Create_Register_DB(std::map <std::string, std::string> Configs){
  std::cout << "Do Not Have DB File, Now To Create it" << std::endl;
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  char* zErrMsg = 0;
  char* Query = "CREATE TABLE `reg_face` ( `IDS` INTEGER PRIMARY KEY AUTOINCREMENT,  `NAME`  TEXT,  `FEATEXE` BLOB, `SIMD_DOT` REAL);";
  int State = sqlite3_exec(Register_db, Query, callback, 0, &zErrMsg);
  if (State != SQLITE_OK)
  {
    std::cout << "Create DB Fail" << std::endl;
    std::cout << State << std::endl;
    sqlite3_close(Register_db);
    return -1;
  }
  sqlite3_close(Register_db);
  return 0;
}



// 2018/12/6 added by Lynn ===================================
int Register_DB_File(std::map <std::string, std::string>& Configs, std::vector<std::string>& File_Name_List, std::vector<cv::Mat>& IMG_List, std::vector<std::vector<float>> &Feature_List){
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  sqlite3_stmt* stmt;
  int List_Size = File_Name_List.size();
  cv::Mat Gray_IMG;
  int Feature_Vector_Size = atoi(Configs["Feature_Vector_Size"].c_str());

  //Lynn, 2018/03/31, to check the register data
  FVV AllMembers;
  Select_Registed_Data(Register_DB_File.c_str(), AllMembers);
  boost::progress_display show_progress( List_Size );
  for (int i = 0 ; i < List_Size ; i++)
  {
    if(Compare_Face_From_DB(AllMembers, 0.99999, &Feature_List[i][0]) >=0 ) // same feature vector
    {
      ++show_progress;
      continue;
    }
    sqlite3_prepare_v2(Register_db, "INSERT INTO `reg_face` (`NAME`,`FEATEXE`,`SIMD_DOT`) values (?,?,?);", -1, &stmt, 0);
    std::vector<std::string> IMG_Path_tmp = split_string(File_Name_List[i], "/");
    std::string toRegistName = IMG_Path_tmp[IMG_Path_tmp.size() - 2];
    float Feature_SIMD_DOT = sqrt(simd_dot(&Feature_List[i][0], &Feature_List[i][0], Feature_Vector_Size));

    sqlite3_bind_text(stmt, 1, toRegistName.c_str(), strlen( toRegistName.c_str()), 0);
    sqlite3_bind_blob(stmt, 2, &Feature_List[i][0], Feature_Vector_Size * sizeof(float), 0);
    sqlite3_bind_double(stmt, 3, Feature_SIMD_DOT);
    sqlite3_step(stmt);
    sqlite3_reset(stmt);
    ++show_progress;
  }
  sqlite3_finalize(stmt);
  sqlite3_close(Register_db);
  return 0;
}
//============================================================



int Register_DB_File(std::map <std::string, std::string>& Configs, std::vector<std::string>& File_Name_List, std::vector<cv::Mat>& IMG_List, int Feature_Vector_Size){
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  sqlite3_stmt* stmt;
  int List_Size = File_Name_List.size();
  cv::Mat Gray_IMG;
  float* Feature = new float[Feature_Vector_Size];

  //Lynn, 2018/03/31, to check the register data
  FVV AllMembers;
  Select_Registed_Data(Register_DB_File.c_str(), AllMembers);

  BufferFile json_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".json");
  BufferFile param_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return -1;
  }

  PredictorHandle pred_hnd = Feature_Net(json_data, param_data, atoi(Configs["Extract_IMG_Size"].c_str()), (char*)Configs["Feature_Layer"].c_str());

  boost::progress_display show_progress( List_Size );
  for (int i = 0 ; i < List_Size ; i++)
  {
    if(IMG_List[i].channels() != 1)
      cv::cvtColor(IMG_List[i], Gray_IMG, CV_BGR2GRAY);
    else
      Gray_IMG = IMG_List[i];
    if(Gray_IMG.cols != 128 && Gray_IMG.rows != 128)
      cv::resize(Gray_IMG, Gray_IMG, cv::Size(128,128));
  	Feature_Extract_exe(Gray_IMG, Feature, pred_hnd);
    
    if(Compare_Face_From_DB(AllMembers, 0.999999, Feature) >=0 ) // same feature vector
    {
      ++show_progress;
      continue;
    }

    sqlite3_prepare_v2(Register_db, "INSERT INTO `reg_face` (`NAME`,`FEATEXE`,`SIMD_DOT`) values (?,?,?);", -1, &stmt, 0);
    
    std::vector<std::string> IMG_Path_tmp = split_string(File_Name_List[i], "/");
    std::string toRegistName = IMG_Path_tmp[IMG_Path_tmp.size() - 2];
    float Feature_SIMD_DOT = sqrt(simd_dot(Feature, Feature, Feature_Vector_Size));
    
    sqlite3_bind_text(stmt, 1, toRegistName.c_str(), strlen( toRegistName.c_str()), 0);
    sqlite3_bind_blob(stmt, 2, Feature, Feature_Vector_Size * sizeof(float), 0);
    sqlite3_bind_double(stmt, 3, Feature_SIMD_DOT);

    sqlite3_step(stmt);
    sqlite3_reset(stmt);
    ++show_progress;
  }
  MXPredFree(pred_hnd);
  sqlite3_finalize(stmt);
  sqlite3_close(Register_db);

  return 0;
}

int Register_DB(std::map <std::string, std::string>& Configs, std::vector<std::string>& File_Name_List, std::vector<cv::Mat>& IMG_List, std::vector<std::vector<float>>& FeatureVec)
{
  int Feature_Vector_Size = FeatureVec[0].size();
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  sqlite3_stmt* stmt;
  int List_Size = File_Name_List.size();
  cv::Mat Gray_IMG;
  //float* Feature = new float[Feature_Vector_Size];

  //Lynn, 2018/03/31, to check the register data
  FVV AllMembers;
  Select_Registed_Data(Register_DB_File.c_str(), AllMembers);

  BufferFile json_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".json");
  BufferFile param_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return -1;
  }

  //PredictorHandle pred_hnd = Feature_Net(json_data, param_data, atoi(Configs["Extract_IMG_Size"].c_str()), (char*)Configs["Feature_Layer"].c_str());

  boost::progress_display show_progress( List_Size );
  for (int i = 0 ; i < List_Size ; i++)
  {
  //  if(IMG_List[i].channels() != 1)
  //    cv::cvtColor(IMG_List[i], Gray_IMG, CV_BGR2GRAY);
  //  else
  //    Gray_IMG = IMG_List[i];
  //  if(Gray_IMG.cols != 128 && Gray_IMG.rows != 128)
  //    cv::resize(Gray_IMG, Gray_IMG, cv::Size(128,128));
  //  Feature_Extract_exe(Gray_IMG, Feature, pred_hnd);
    
    if(Compare_Face_From_DB(AllMembers, 0.999999, &FeatureVec[i][0]) >=0 ) // same feature vector
    {
        ++show_progress;
        continue;
    }

    sqlite3_prepare_v2(Register_db, "INSERT INTO `reg_face` (`NAME`,`FEATEXE`,`SIMD_DOT`) values (?,?,?);", -1, &stmt, 0);
    
    std::vector<std::string> IMG_Path_tmp = split_string(File_Name_List[i], "/");
    std::string toRegistName = IMG_Path_tmp[IMG_Path_tmp.size() - 2];
    float Feature_SIMD_DOT = sqrt(simd_dot(&FeatureVec[i][0], &FeatureVec[i][0], Feature_Vector_Size));
    
    sqlite3_bind_text(stmt, 1, toRegistName.c_str(), strlen( toRegistName.c_str()), 0);
    sqlite3_bind_blob(stmt, 2, &FeatureVec[i][0], Feature_Vector_Size * sizeof(float), 0);
    sqlite3_bind_double(stmt, 3, Feature_SIMD_DOT);

    sqlite3_step(stmt);
    sqlite3_reset(stmt);
    ++show_progress;
  }
  //MXPredFree(pred_hnd);
  sqlite3_finalize(stmt);
  sqlite3_close(Register_db);

  return 0;
}


int Register_and_Check_DB_File(std::map <std::string, std::string> Configs, std::vector<std::string> File_Name_List, std::vector<cv::Mat> IMG_List, int Feature_Vector_Size){
  sqlite3* Register_db;
  std::string Register_DB_File = Configs["Register_DB_Name"];
  sqlite3_open(Register_DB_File.c_str(), &Register_db);
  sqlite3_stmt* stmt;
  char* Check;
  int List_Size = File_Name_List.size();
  cv::Mat Gray_IMG;
  float* Feature = new float[Feature_Vector_Size];
  std::string Query;

  BufferFile json_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".json");
  BufferFile param_data(Configs["Model_Dir"] + Configs["Model_Name"] + ".params");
  if(json_data.GetLength() == 0 || param_data.GetLength() == 0){
    return -1;
  }
  
  PredictorHandle pred_hnd = Feature_Net(json_data, param_data, atoi(Configs["Extract_IMG_Size"].c_str()), (char*)Configs["Feature_Layer"].c_str());

  boost::progress_display show_progress( List_Size );
  for (int i = 0 ; i < List_Size ; i++)
  {
    Query = "SELECT * FROM  `reg_face` Where `NAME`=\"" + File_Name_List[i] + "\";";
    sqlite3_prepare_v2(Register_db, Query.c_str(), -1, &stmt, 0);
    sqlite3_step(stmt);
    Check = (char*)sqlite3_column_text( stmt, 0 );
    float *Feature_tmp = (float*)sqlite3_column_blob( stmt, 2 );
    int test_int = (int )sqlite3_column_bytes( stmt, 2 );
    sqlite3_reset(stmt);
      
    if (Check == NULL || Check == " ")
    {
      if(IMG_List[i].channels() != 1)
        cv::cvtColor(IMG_List[i], Gray_IMG, CV_BGR2GRAY);
      else
        Gray_IMG = IMG_List[i];
      cv::resize(Gray_IMG, Gray_IMG, cv::Size(128,128));
  		Feature_Extract_exe(Gray_IMG, Feature, pred_hnd);
      float Feature_SIMD_DOT = sqrt(simd_dot(Feature, Feature, Feature_Vector_Size));
      sqlite3_prepare_v2(Register_db, "INSERT INTO `reg_face` (`NAME`,`FEATEXE`,`SIMD_DOT`) values (?,?,?);", -1, &stmt, 0);
      sqlite3_bind_text(stmt, 1, File_Name_List[i].c_str(), strlen( File_Name_List[i].c_str()), 0);
      sqlite3_bind_blob(stmt, 2, Feature, Feature_Vector_Size * sizeof(float), 0);
      sqlite3_bind_double(stmt, 3, Feature_SIMD_DOT);
      sqlite3_step(stmt);
      sqlite3_reset(stmt);
    }
    ++show_progress;
  }
  MXPredFree(pred_hnd);
  sqlite3_finalize(stmt);
  sqlite3_close(Register_db);

  return 0;
}










//------------------------ 2018/09/09 , Lynn
bool Select_Registed_Data(PDB_Face& dbc, P_FV &Mbs)
{
  FVecV tmp_empty;

  Mbs.clear();
  tmp_empty.clear();

  // if(!dbc.select_data("select * from person;"))
  if(!dbc.select_data("select * from Valid_person order by pid;"))
  {
    fprintf(stderr, "Select Person error.\n");
    return false;
  }


  // FILE *out = fopen("queryIDnumber2.txt", "wt");


  for(result::const_iterator i = dbc.query_result->begin(); i != dbc.query_result->end() ; i++)
  {
    binarystring b_name(i["NAME"]);
    binarystring b_RoleT(i["RoleTitle"]);
    std::string s_name = b_name.str();
    std::string s_role = b_RoleT.str();
    std::string s_mail = i["Email"].as<std::string>();
    std::string s_studID = i["StudentID"].as<std::string>();
    std::string s_cdID = i["CardID"].as<std::string>();
    std::string s_pimg = i["ProfileImg"].as<std::string>();

    Person tmp_p;
    tmp_p.pid = i["PID"].as<int>();
    tmp_p.pflag = i["personflag"].as<int>();
    std::strcpy(tmp_p.name, s_name.c_str());
    std::strcpy(tmp_p.title, s_role.c_str());
    std::strcpy(tmp_p.email, s_mail.c_str());
    std::strcpy(tmp_p.student_id, s_studID.c_str());
    std::strcpy(tmp_p.card_id, s_cdID.c_str());
    std::strcpy(tmp_p.profile_img, s_pimg.c_str());



    // fprintf(out, "pid = %d\n", tmp_p.pid);
    // fprintf(out, "name = %s\n", tmp_p.name);
    // fprintf(out, "title = %s\n", tmp_p.title);
    // fprintf(out, "email = %s\n", tmp_p.email);
    // fprintf(out, "id = %s\n", tmp_p.student_id);
    // fprintf(out, "card_id = %s\n\n", tmp_p.card_id);


    
    Mbs[tmp_p] = tmp_empty;
  }

  
  // fclose(out);


  // if(!dbc.select_data("select f.FID, f.PID, f.Simd_Dot, f.Img_Path, f.Feature, p.StudentID from Face_Data f, Person p where f.PID = p.PID;"))
  if(!dbc.select_data("select * from Valid_Face;"))
  {
    Mbs.clear();
    fprintf(stderr, "Select feature data error.\n");
    return false;
  }
  for(result::const_iterator i = dbc.query_result->begin(); i != dbc.query_result->end() ; i++)
  {
    FeatureVec tmp_fv;
    tmp_fv.fid = i[0].as<int>();
    tmp_fv.sqrt_simdot = i[2].as<float>();
    std::strcpy(tmp_fv.img_path, i[3].c_str());

    binarystring sel_fv(i[4]);
    float *db_fv = (float*)((void*)sel_fv.data());
    std::memcpy(tmp_fv.fv, db_fv, sizeof(float)*fvSize);

    Person tmp_p;
    tmp_p.pid = i[1].as<int>();
    std::strcpy(tmp_p.student_id, i[5].c_str());
    Mbs[tmp_p].push_back(tmp_fv);
  }

  return true;
}

bool Compare_Face_Person(const FVecV &Mbs, float sim_th, float* Feature_Vector, float& maxsim, bool is_cal_dot = false, float cal_dot = 0)
{
  bool is_this_person = false;
  float Feature_simd_dot, sim;

  if(is_cal_dot)Feature_simd_dot = cal_dot;
  else Feature_simd_dot = sqrt(simd_dot(Feature_Vector, Feature_Vector, fvSize));

  int all_size = Mbs.size();
  #pragma omp parallel for
  for (int xi=0;xi<all_size;xi++)
  {
    sim = NAN;
    sim = simd_dot(Mbs[xi].fv, Feature_Vector, fvSize) / (Mbs[xi].sqrt_simdot * Feature_simd_dot);
    if (std::isnan(sim)) // no reg data
      continue;

    #pragma omp critical
    {
      if(sim >= sim_th && sim > maxsim)
      {
        maxsim = sim;
        is_this_person = true;
      }
    }
  }
  return is_this_person;
}

Person Compare_Face_DB(P_FV &Mbs, float sim_th, float* Feature_Vector)
{
  float Feature_simd_dot = sqrt(simd_dot(Feature_Vector, Feature_Vector, fvSize));
  float sim , maxsim=0;
  int all_size = Mbs.size() , rtidx = -1;
  Person the_most_like;

  for(P_FV::iterator i = Mbs.begin(); i!=Mbs.end(); i++)
    if(Compare_Face_Person(i->second,sim_th,Feature_Vector,maxsim,true,Feature_simd_dot))
      the_most_like = i->first;

  return the_most_like;
}

void Register_PDB(PDB_Face &dbc, P_FV &Mbs, PredictorHandle &pred_hnd)
{
  for(P_FV::iterator i = Mbs.begin(); i!= Mbs.end(); i++)
  {
    const Person& _p = i->first;

    // default save to not_verify
    char the_person_dir[FILENAME_MAX], the_img_path[FILENAME_MAX], tmp_to_access_img[FILENAME_MAX];
    sprintf(the_person_dir,"not_verify/%s/", _p.student_id);
    sprintf(tmp_to_access_img,"%s%s",face_db_data_root,the_person_dir);
    chkDir(tmp_to_access_img);

    // there is a bug
    // if the Person data is read from database,
    // the perfile image must concat the face_db_data_root.
    // the code now can only handle the condition that
    // the profile image is save to the directory that the program can read it directly.
    cv::Mat prof_img = cv::imread(_p.profile_img);

    sprintf(the_img_path,"%sprofile.png",the_person_dir);
    sprintf(tmp_to_access_img,"%s%s",face_db_data_root,the_img_path);
    // once the image is saved to database, the path in the database
    // is relate to face_db_data_root.
    cv::imwrite(tmp_to_access_img, prof_img);

    int pid = dbc.insert_person(_p.name, _p.title, _p.email, _p.student_id, _p.card_id, _p.pflag, the_img_path);
    if(pid < 0)
    {
      fprintf(stderr, "%s is registered fail.\n", _p.student_id);
      continue;
    }

    for(auto& pfv : i->second)
    {
      if(!pfv.I.data)continue;

      cv::Mat IMG_GRAY = pfv.I.clone();
      if(IMG_GRAY.channels() == 3)
        cv::cvtColor(IMG_GRAY, IMG_GRAY, CV_BGR2GRAY);
      Feature_Extract_exe(IMG_GRAY, pfv.fv, pred_hnd);
      pfv.sqrt_simdot = sqrt(simd_dot(pfv.fv, pfv.fv, fvSize));

      int fid = dbc.insert_face_data(pid, pfv.sqrt_simdot, the_person_dir, pfv.fv);
      if(fid < 0)
      {
        fprintf(stderr, "%s: Feature insert error.\n", _p.student_id);
        continue;
      }
      sprintf(pfv.img_path, "%s%d.png", the_person_dir, fid);
      sprintf(tmp_to_access_img,"%s%s",face_db_data_root,pfv.img_path);
      cv::imwrite(tmp_to_access_img, pfv.I);
      dbc.update_face_imgpath(fid, pfv.img_path);
    }
  }
}
void Register_PDB_onlyCard(PDB_Face &dbc, long regCardID, FVecV &lstimg, PredictorHandle &pred_hnd)
{
  char regcard[30];
  sprintf(regcard,"%ld",regCardID);
  int tmp_pid = dbc.get_tmpReg_pid();
  int rid = dbc.insert_wanna_regist(regcard);

  // the temp root is "regist_list"
  char tmp_reg_dir[FILENAME_MAX], tmp_to_access_img[FILENAME_MAX];
  sprintf(tmp_reg_dir,"regist_list/%d/", rid);
  sprintf(tmp_to_access_img,"%s%s",face_db_data_root,tmp_reg_dir);
  chkDir(tmp_to_access_img);

  for(auto& pfv : lstimg)
  {
    if(!pfv.I.data)continue;

    cv::Mat IMG_GRAY = pfv.I.clone();
    if(IMG_GRAY.channels() == 3)
      cv::cvtColor(IMG_GRAY, IMG_GRAY, CV_BGR2GRAY);
    Feature_Extract_exe(IMG_GRAY, pfv.fv, pred_hnd);
    pfv.sqrt_simdot = sqrt(simd_dot(pfv.fv, pfv.fv, fvSize));

    int fid = dbc.insert_face_data(tmp_pid, pfv.sqrt_simdot, "TMP", pfv.fv);
    if(fid < 0)
    {
      fprintf(stderr, "Pre-regist temp Feature insert error.\n");
      continue;
    }
    sprintf(pfv.img_path, "%s%d.png", tmp_reg_dir, fid);
    sprintf(tmp_to_access_img,"%s%s",face_db_data_root,pfv.img_path);
    cv::imwrite(tmp_to_access_img, pfv.I);
    dbc.update_face_imgpath(fid, pfv.img_path);
    dbc.insert_RF(rid, fid);
  }
}

// return PID
bool isExistIDNumber(P_FV &Mbs, Person& get_person, long IDNumber)
{
  // FILE *out = fopen("queryIDnumber.txt", "wt");
  // fprintf(out, "# of registered person: %d\n", Mbs.size());

  get_person.pid = -1;
  for(P_FV::iterator i = Mbs.begin(); i!= Mbs.end(); i++)
  {
    const Person& _p = i->first;

    // fprintf(out, "pid = %d\n", _p.pid);
    // fprintf(out, "name = %s\n", _p.name);
    // fprintf(out, "title = %s\n", _p.title);
    // fprintf(out, "email = %s\n", _p.email);
    // fprintf(out, "id = %s\n", _p.student_id);
    // fprintf(out, "card_id = %s\n", _p.card_id);

    long _psid = atol(_p.student_id);

    // fprintf(out, "conderted ID = %lu\n", _psid);

    if(_psid == IDNumber)
    {
      get_person = _p;
      // fprintf(out, "Matched!\n");
      // fclose(out);

      return true;
    }
  }
  // fprintf(out, "Unmatched!\n");
  // fclose(out);
  return false;
}
//------------------------ 2018/09/09 , Lynn