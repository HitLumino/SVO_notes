// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {
//InitResult:初始化返回结果 FAILURE, NO_KEYFRAME, SUCCESS
//! 只会辨别特征点的数量是否大于100,无法进行光流跟踪
//! 1. reset();特征点/frame_ref_参考帧 清零
//! 2. 检测特征点
//! 3. 获取特征点的像素坐标以及空间点的归一化坐标
//! 4. 检测特征点数量如果小于100,返回失败;若大于100,将此帧设为参考帧(frame_ref_)
//! 5. 将当前帧设为参考帧;并将参考帧的特征点坐标(px_ref_)赋值给当前帧的特征点(px_cur_)
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  // 特征点的坐标为原始图像（金字塔第0层）的像素坐标
  detectFeatures(frame_ref, px_ref_, f_ref_);///获取特征点的像素坐标以及空间点的归一化坐标
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());//将参考帧的特征点坐标赋值给当前帧的特征点,从begin()之前开始
  return SUCCESS;
}

/* KLT跟踪算法
 * http://blog.sina.com.cn/s/blog_4f3a049701014w91.html
 */
/**
 * @brief KltHomographyInit::addSecondFrame
 * @param frame_cur
 * 1. KLT跟踪算法得到第二帧的特征点坐标px_cur_  f_cur_ 以及 disparities_
 * 2. 根据光流跟踪的结果计算单应性矩阵,  顺带计算出来了特征点的深度xyz_in_cur_
 * 3. 算出地图的 scene_depth_median -->算出尺度scale=Config::mapScale()/scene_depth_median;//Config::mapScale()=1
 * 4. For each inlier create 3D point and add feature in both frames
 * @return
 */
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
    // 第二帧图像并没有提取特征点
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  // computeHomography 顺带计算出来了特征点的深度xyz_in_cur_
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);
  double scale = Config::mapScale()/scene_depth_median; // 因为computeHomography 算出来的平移是带尺度的，要归一化到统一的尺度下
  // 当前坐标系的位姿
  // quaternion: [0, 0, 0, 1],  translation:[0, 0, 0]
  // 左乘矩阵：说明为点坐标变换矩阵
  //T_cur_from_ref_: 由computeHomography函数得出
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;//frame_ref_->T_f_w_为单位矩阵
  ///为了计算 SE3 T_world_cur 所以将平移向量逆处理
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  /// For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();///?????对不对 直接取逆
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur); // 特征点也要存好它被哪些帧看到了。bundler adjustment 的时候要用

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

/// 检测特征点函数
//! 1. 生成对象FastDetector detector
//! 2. detect()函数,返回Features new_features(vector)
//! 3. 获取特征点的坐标-->px_vec
//! 4. 获取3d归一化坐标-->f_vec   f = [(u-ox)/fx , (v-oy)/fy , 1]
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // 根据生成特征点的数量
  // 获取特征点的坐标-->px_vec
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));///获取特征点的坐标
    // 获取3d坐标-->f_vec
    f_vec.push_back(ftr->f);/// f = [(u-ox)/fx , (v-oy)/fy , 1] ;  f * depth --> [x , y , z] : 3d 坐标in this frame
    delete ftr;
  });
}
/**
 * 根据第一帧提取的fast角点,用光流法跟踪 主要函数为 calcOpticalFlowPyrLK
 * @param frame_ref    参考帧(上一帧)
 * @param frame_cur    当前帧
 * @param px_ref       参考帧的特征点的2d像素坐标(原来的像素)
 * @param px_cur       光流计算的在当前帧中的像素坐标                                                        <输出>
 * @param f_ref        参考帧的特征点的相机坐标系下的3d坐标
 * @param f_cur        当前帧的特征点的相机坐标系下的3d坐标                                               <输出>
 * @param disparities  视差向量Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm()  <输出>
 * 光流跟踪函数原型
 * CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                        InputArray prevPts, InputOutputArray nextPts,
                                        OutputArray status, OutputArray err,
                                        Size winSize = Size(21,21), int maxLevel = 3,
                                        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                        int flags = 0, double minEigThreshold = 1e-4 );
 * @param maxLevel  图像金字塔的层数,此处为4   0:默认不采用金字塔
 * @param termcrit  迭代终止条件
 * **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is
     not set, then prevPts is copied to nextPts and is considered the initial estimate.
 */

void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  /// Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  //计算当前帧中特征点在当前帧相机坐标系中的方向
  f_cur.clear(); f_cur.reserve(px_cur.size());
  //计算光流法得到的视差
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    if(!status[i])//根据状态剔除坏点  状态非1 表示没有找到匹配点
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));//c2f 像素坐标系-->相机坐标系
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

/**
 * @brief computeHomography            计算单应矩阵
 * @param f_ref
 * @param f_cur
 * @param focal_length
 * @param reprojection_threshold
 * @param inliers
 * @param xyz_in_cur
 * @param T_cur_from_ref
 */
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d, aligned_allocator<Vector2d> > uv_ref(f_ref.size());
  vector<Vector2d, aligned_allocator<Vector2d> > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);///(u,v)
    uv_cur[i] = vk::project2d(f_cur[i]);
  }
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
