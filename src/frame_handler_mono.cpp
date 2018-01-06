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
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
  initialize();
}

void FrameHandlerMono::initialize()
{
   //! typedef boost::shared_ptr<AbstractDetector> ---> DetectorPtr;
   //! feature_detection ��һ�������ռ�,����AbstractDetector
   //! feature_detection::FastDetector�̳���feature_detection::DetectorPtr
   //! boost::arg<1> _1; boost::arg<2> _2;

  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}
//! FrameHandlerMono::addImage() : ͼ������
//! ����״̬�Ĳ�ͬѡ��ͬ�Ĵ�����
//!     1. STAGE_DEFAULT_FRAME --> processFrame() //���㹻��������͹�
//!     2. STAGE_SECOND_FRAME --> processSecondFrame() //��Ҫ�㹻���Ӳ�
//!     3. STAGE_FIRST_FRAME --> processFirstFrame()
//!     4. STAGE_RELOCALIZING --> relocalizeFrame()
void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();//!< Keyframes in the closer neighbourhood.
  overlap_kfs_.clear();//!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?

  /// create new frame
  /// create image_pyramid
  //#define SVO_START_TIMER(name) g_permon->startTimer((name))
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));//!new_frame_:  Current frame.
  //#define SVO_STOP_TIMER(name) g_permon->stopTimer((name))
  SVO_STOP_TIMER("pyramid_creation");

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));

  // set last frame
  last_frame_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
}

//! �����һ֡ͼ��processFirstFrame()
//! 1. ��ǰ֡��Ϊ��λ���� SE3(Matrix3d::Identity(), Vector3d::Zero())
//! 2. �ж�klt_homography_init_ : ͨ�������� ǰ��֡�ָ����ĵ�Ӧ�Ծ���
//! 3. ����ǰ֡��Ϊ�ؼ�֡ new_frame_->setKeyframe();
//!     * setKeyPoints();
//!         * checkKeyPoints()
//! 4. ��ͼ����Ӹùؼ�֡ map_.addKeyframe(new_frame_);
//! 5. ״̬��ΪSTAGE_SECOND_FRAME
//! 6. ���� RESULT_IS_KEYFRAME
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

//! processSecondFrame()���ش�����,��/��/ʧ��
//! 1. klt_homography_init_.addSecondFrame(new_frame_);
//!     * addSecondFrame()������������ù��������ٵ�һ֡������������H ����--> ���λ�ˣ��������������
//! 2. new_frame_->setKeyframe();
//! 3. ��ȡƽ�����-->����˲�����Ӹùؼ�֡
//! 4. ��ͼ��ӹؼ�֡
//! 5. ״̬��ΪSTAGE_DEFAULT_FRAME
//! 6. ����RESULT_IS_KEYFRAME
FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  //addSecondFrame() ������������ù��������ٵ�һ֡������������H ����--> ���λ�ˣ��������������
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_); 
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

//! ������ͨ�ؼ�֡(�ǳ�ʼ��ǰ��֡)
//! ���ش�����,��/��/ʧ��
//! processFrame()����ͼ��֡ ���ش����� �ǹؼ�֡/��/ʧ��
//! 1. ���ó�ʼλ��
//! 2. sparse image align
//!    * SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),30, SparseImgAlign::GaussNewton, false, false);
//!    * size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
//! 3. ��ͼ�еĵ���ͶӰ & feature alignment
//!    * reprojectMap-->reprojectCell-->findMatchDirect ����feature alignment
//!    * �ж����������Ƿ�config����ֵ?
//!         * [-y]-->λ�˻�����Ϊ��һ�ε�λ��-->��������ΪTRACKING_INSUFFICIENT-->���ؽ��Ϊʧ��
//! 4. λ���Ż� (��˹ţ�� ��ͶӰλ�����)
//!    * pose_optimizer::optimizeGaussNewton(Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
//!    new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
//!        * sfba_n_edges_final(�Ż���ıߵ������Ƿ�С��20) [-y] ����ʧ��
//! 5. ����Ż� (point_optimizer)
//!    * optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
//! 6. select keyframe
//!     *��ӵ�core_kfs_(���Ĺؼ�֡)
//!     * setTrackingQuality(sfba_n_edges_final)������������
//!     * KF ���жϱ�׼: ���new_frame_ ���������ڵ�����KF֮������ƽ�ƶ������˳���ƽ����ȵ�12%
//!         * ����: �������KF,�Ǿ������������ֵ return RESULT_NO_KEYFRAME;
//!         * ��: ����KF
//!     *���ùؼ�֡ new_frame_->setKeyframe();
//! 7. ����˲�����ӹؼ�֡ depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
//!     * if limited number of keyframes, remove the one furthest apart
//! 8. ��ͼ����ӹؼ�֡ map_.addKeyframe(new_frame_);
//!
FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
  // Set initial pose TODO use prior
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  // ����ͼ�еĵ���ͶӰ��new_frame, ��������aligment. 
  // reprojectMap-->reprojectCell-->findMatchDirect ����feature alignment
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  //�ж����������Ƿ�config����ֵ
  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;
    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  // ����λ�˱任������Ż���ͶӰλ�����
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  // ������ά������(x,y,z)�Ż���ͶӰλ�����
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    return RESULT_FAILURE;
  }
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  // KF ���жϱ�׼: ���new_frame_ ���������ڵ�����KF֮������ƽ�ƶ������˳���ƽ����ȵ�12%
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_); // �������KF,�Ǿ������������ֵ
    return RESULT_NO_KEYFRAME;
  }
  // ����KF
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);//

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);

  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}
//Keyframe selection criterion �ؼ�֡��ѡ����
bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
    //vector< pair<FramePtr,size_t> > overlap_kfs_; size_t:�ж��ٵ�ͼ���ǹ��ӵ�
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
