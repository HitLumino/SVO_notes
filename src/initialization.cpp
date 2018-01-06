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
//InitResult:��ʼ�����ؽ�� FAILURE, NO_KEYFRAME, SUCCESS
//! ֻ����������������Ƿ����100,�޷����й�������
//! 1. reset();������/frame_ref_�ο�֡ ����
//! 2. ���������
//! 3. ��ȡ����������������Լ��ռ��Ĺ�һ������
//! 4. ����������������С��100,����ʧ��;������100,����֡��Ϊ�ο�֡(frame_ref_)
//! 5. ����ǰ֡��Ϊ�ο�֡;�����ο�֡������������(px_ref_)��ֵ����ǰ֡��������(px_cur_)
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  // �����������Ϊԭʼͼ�񣨽�������0�㣩����������
  detectFeatures(frame_ref, px_ref_, f_ref_);///��ȡ����������������Լ��ռ��Ĺ�һ������
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());//���ο�֡�����������긳ֵ����ǰ֡��������,��begin()֮ǰ��ʼ
  return SUCCESS;
}

/* KLT�����㷨
 * http://blog.sina.com.cn/s/blog_4f3a049701014w91.html
 */
/**
 * @brief KltHomographyInit::addSecondFrame
 * @param frame_cur
 * 1. KLT�����㷨�õ��ڶ�֡������������px_cur_  f_cur_ �Լ� disparities_
 * 2. ���ݹ������ٵĽ�����㵥Ӧ�Ծ���,  ˳���������������������xyz_in_cur_
 * 3. �����ͼ�� scene_depth_median -->����߶�scale=Config::mapScale()/scene_depth_median;//Config::mapScale()=1
 * 4. For each inlier create 3D point and add feature in both frames
 * @return
 */
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
    // �ڶ�֡ͼ��û����ȡ������
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  // computeHomography ˳���������������������xyz_in_cur_
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
  double scale = Config::mapScale()/scene_depth_median; // ��ΪcomputeHomography �������ƽ���Ǵ��߶ȵģ�Ҫ��һ����ͳһ�ĳ߶���
  // ��ǰ����ϵ��λ��
  // quaternion: [0, 0, 0, 1],  translation:[0, 0, 0]
  // ��˾���˵��Ϊ������任����
  //T_cur_from_ref_: ��computeHomography�����ó�
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;//frame_ref_->T_f_w_Ϊ��λ����
  ///Ϊ�˼��� SE3 T_world_cur ���Խ�ƽ�������洦��
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  /// For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();///?????�Բ��� ֱ��ȡ��
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
      new_point->addFrameRef(ftr_cur); // ������ҲҪ���������Щ֡�����ˡ�bundler adjustment ��ʱ��Ҫ��

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

/// ��������㺯��
//! 1. ���ɶ���FastDetector detector
//! 2. detect()����,����Features new_features(vector)
//! 3. ��ȡ�����������-->px_vec
//! 4. ��ȡ3d��һ������-->f_vec   f = [(u-ox)/fx , (v-oy)/fy , 1]
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
  // �������������������
  // ��ȡ�����������-->px_vec
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));///��ȡ�����������
    // ��ȡ3d����-->f_vec
    f_vec.push_back(ftr->f);/// f = [(u-ox)/fx , (v-oy)/fy , 1] ;  f * depth --> [x , y , z] : 3d ����in this frame
    delete ftr;
  });
}
/**
 * ���ݵ�һ֡��ȡ��fast�ǵ�,�ù��������� ��Ҫ����Ϊ calcOpticalFlowPyrLK
 * @param frame_ref    �ο�֡(��һ֡)
 * @param frame_cur    ��ǰ֡
 * @param px_ref       �ο�֡���������2d��������(ԭ��������)
 * @param px_cur       ����������ڵ�ǰ֡�е���������                                                        <���>
 * @param f_ref        �ο�֡����������������ϵ�µ�3d����
 * @param f_cur        ��ǰ֡����������������ϵ�µ�3d����                                               <���>
 * @param disparities  �Ӳ�����Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm()  <���>
 * �������ٺ���ԭ��
 * CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                        InputArray prevPts, InputOutputArray nextPts,
                                        OutputArray status, OutputArray err,
                                        Size winSize = Size(21,21), int maxLevel = 3,
                                        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                        int flags = 0, double minEigThreshold = 1e-4 );
 * @param maxLevel  ͼ��������Ĳ���,�˴�Ϊ4   0:Ĭ�ϲ����ý�����
 * @param termcrit  ������ֹ����
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
  //���㵱ǰ֡���������ڵ�ǰ֡�������ϵ�еķ���
  f_cur.clear(); f_cur.reserve(px_cur.size());
  //����������õ����Ӳ�
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    if(!status[i])//����״̬�޳�����  ״̬��1 ��ʾû���ҵ�ƥ���
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));//c2f ��������ϵ-->�������ϵ
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

/**
 * @brief computeHomography            ���㵥Ӧ����
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
