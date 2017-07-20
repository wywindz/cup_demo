#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT> CloudNormalT;

std::string topicName;
bool updateFlag;
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr RecvRawData(new pcl::PointCloud<pcl::PointXYZRGBA>);
PointCloudType::Ptr recvCloud(new PointCloudType());
PointCloudType::Ptr filteredCloud(new PointCloudType());
CloudNormalT::Ptr cloudNormals(new CloudNormalT);
PointCloudType::Ptr baseCylinder(new PointCloudType);
PointCloudType::Ptr alignedCylinder(new PointCloudType);
Eigen::Matrix4f transformation(Eigen::Matrix4f::Identity());
float leafSize;
double passXmin,passXmax,passYmin,passYmax,passZmin,passZmax;
bool initalAcquire;

void initGlobalParas(void)
{
  topicName="UER_K2_TOPIC_CLOUDXYZRGBA";
  transformation=Eigen::Matrix4f::Identity();
  leafSize=0.001;
  passZmin=0.0;
  passZmax=0.8;
  passXmin=-0.5;
  passXmax=0.5;
  passYmin=-0.5;
  passYmax=0.5;
  updateFlag=false;
  initalAcquire=true;

}


void cloudProcess(PointCloudType::Ptr sourceCloud)
{

  pcl::PointXYZ minPtr,maxPtr;
  //pcl::getMinMax3D(*sourceCloud,minPtr,maxPtr);
  /*
  std::cout<<"MinX: "<<minPtr.x<<std::endl;
  std::cout<<"MinY: "<<minPtr.y<<std::endl;
  std::cout<<"MinZ: "<<minPtr.z<<std::endl;
  std::cout<<"MaxX: "<<maxPtr.x<<std::endl;
  std::cout<<"MaxY: "<<maxPtr.y<<std::endl;
  std::cout<<"MaxZ: "<<maxPtr.z<<std::endl;
  */
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(sourceCloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(passZmin,passZmax);
  pass.filter(*filteredCloud);

  pass.setInputCloud(filteredCloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(passXmin,passXmax);
  pass.filter(*filteredCloud);

  pass.setInputCloud(filteredCloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(passYmin,passYmax);
  pass.filter(*filteredCloud);

  //Down Sample
  grid.setLeafSize(leafSize,leafSize,leafSize);
  grid.setInputCloud(filteredCloud);
  grid.filter(*filteredCloud);
  
  //remove outliers
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(filteredCloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1);
  sor.filter(*filteredCloud);

  std::cout<<std::endl;
  std::cout<<"================================================="<<std::endl;
  std::cout<<"==> Downsample Processed, Point size is: "<<filteredCloud->points.size()<<std::endl;
}

PointCloudType::Ptr segmentateCylinder(PointCloudType::Ptr sceneCloud)
{
  PointCloudType::Ptr cylinderCloud(new PointCloudType);
  //Compute Normals
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
  pcl::NormalEstimation<PointType,pcl::Normal> ne;
  ne.setSearchMethod(tree);
  ne.setInputCloud(sceneCloud);
  ne.setKSearch(50);
  ne.compute(*cloudNormals);

  //Segment plane
  pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
  pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);

  pcl::SACSegmentationFromNormals<PointType,NormalT> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight(0.1);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.03);
  seg.setInputCloud(sceneCloud);
  seg.setInputNormals(cloudNormals);
  seg.segment(*inliers_plane,*coefficients_plane);
  //std::cout<<"segment plane, coefficients: "<<*coefficients_plane<<std::endl;
  
  //extract
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(sceneCloud);
  extract.setIndices(inliers_plane);
  extract.setNegative(true);
  PointCloudType::Ptr cloud_withoutPlane(new PointCloudType);
  extract.filter(*cloud_withoutPlane);
  std::cout<<"==> Extract plane, point size is: "<<cloud_withoutPlane->points.size()<<std::endl;

  /*
  pcl::ExtractIndices<NormalT> extract_normal;
  extract_normal.setInputCloud(cloudNormals);
  extract_normal.setIndices(inliers_plane);
  extract_normal.setNegative(true);*/
  CloudNormalT::Ptr cloudNormal_withoutPlane(new CloudNormalT);
  //extract_normal.filter(*cloudNormal_withoutPlane);
  ne.setInputCloud(cloud_withoutPlane);
  ne.compute(*cloudNormal_withoutPlane);

  //segment cylinder
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(500);
  seg.setDistanceThreshold(0.05);
  seg.setRadiusLimits(0,0.1);
  seg.setInputCloud(cloud_withoutPlane);
  seg.setInputNormals(cloudNormal_withoutPlane);
  seg.segment(*inliers_cylinder,*coefficients_cylinder);
  //std::cout<<"==> Segment cylinder, coefficients: "<<*coefficients_cylinder<<std::endl;

  extract.setInputCloud(cloud_withoutPlane);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  extract.filter(*cylinderCloud);
  if(cylinderCloud->points.empty())
  {
    std::cerr<<"!!! Can't find cylindrical component."<<std::endl;
  }

  std::cout<<"==> Find cylinder object, point size is: "<<cylinderCloud->points.size()<<std::endl;

  return cylinderCloud;
}


void poseEstimate(PointCloudType::Ptr targetCloud, PointCloudType::Ptr templateCloud)
{
  pcl::IterativeClosestPoint<PointType,PointType> icp;
  icp.setMaximumIterations(200);
  icp.setInputSource(templateCloud);
  icp.setInputTarget(targetCloud);
  icp.align(*alignedCylinder);
  if(icp.hasConverged())
  {
    std::cout<<"==> ICP Alignment has converged."<<std::endl;
    std::cout<<"    Fitness score: "<<icp.getFitnessScore()<<std::endl;
    transformation=icp.getFinalTransformation()*transformation;
    std::cout<<"    transformation matrix: "<<std::endl<<transformation<<std::endl;
  }
  else
  {
    std::cout<<"!!! estimate transformation error!"<<std::endl;
  }
}

void msgCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::fromROSMsg(*msg,*RecvRawData);
  recvCloud->points.resize(RecvRawData->points.size());
  for(size_t i=0;i<RecvRawData->points.size();++i)
  {
    recvCloud->points[i].x=RecvRawData->points[i].x;
    recvCloud->points[i].y=RecvRawData->points[i].y;
    recvCloud->points[i].z=RecvRawData->points[i].z;
  }
  cloudProcess(recvCloud);
  PointCloudType::Ptr cylinder=segmentateCylinder(filteredCloud);
  if(initalAcquire)
    baseCylinder=cylinder;
  else
    baseCylinder=alignedCylinder;//Last pose as the initial guess
  poseEstimate(cylinder,baseCylinder);
  initalAcquire=false;
  updateFlag=true;
}

int
main(int argc, char** argv)
{
  std::cout<<"Cup demo..."<<std::endl;
  initGlobalParas();
  //create subscriber and register callback function
  ros::init(argc,argv,"cup_demo");
  ros::NodeHandle nh("Kinect2_Relay_Msg");
  ros::Subscriber sub=nh.subscribe<sensor_msgs::PointCloud2>(topicName,1000,msgCallback);

  ros::AsyncSpinner spinner(0);
  spinner.start();

  ros::Rate rate(10);
  while(!updateFlag)
  {
    if(!ros::ok())
    {
      return -1;
    }
  }

  pcl::visualization::PCLVisualizer viewer("scene cloud");
  pcl::visualization::PointCloudColorHandlerCustom<PointType> colorHandle(recvCloud,200,200,200);
  viewer.addPointCloud(filteredCloud,colorHandle,"sceneCloud");
  viewer.setBackgroundColor(0,0,0,0);
  //viewer.addCoordinateSystem(1.0);
  viewer.initCameraParameters();
  viewer.setCameraPosition(0,0,0,0,-1,0);

  pcl::visualization::PCLVisualizer viewer_cylinder("cylinder");
  pcl::visualization::PointCloudColorHandlerCustom<PointType> cylinderColorH(alignedCylinder,100,200,100);
  viewer_cylinder.addPointCloud(alignedCylinder,cylinderColorH,"cylinder");
  viewer_cylinder.setBackgroundColor(0,0,0,0);
  viewer_cylinder.initCameraParameters();
  viewer_cylinder.setCameraPosition(0,0,0,0,-1,0);

  while(!viewer.wasStopped())
  {
    if(updateFlag)
    {
      viewer.updatePointCloud(filteredCloud,"sceneCloud");
      viewer_cylinder.updatePointCloud(alignedCylinder,"cylinder");
      updateFlag=false;
    }
    viewer.spinOnce(10);
    viewer_cylinder.spinOnce(10);
    //rate.sleep();
  }

  viewer.close();
  viewer_cylinder.close();
}
