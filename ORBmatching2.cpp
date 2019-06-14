#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

double DIST_THRESHOLD = 30;
int NUM_MATCH_POINTS = 1000;

vector<int> octave1;
vector<int> octave2;
 vector<char> matchesMask;
//ORB特征提取
//1.读取图像
//2.初始化KeyPoint、Descriptor、ORB对象
//3.检测FAST角点
//4.由角点位置计算BRIEF描述子
//5.新建匹配对象及vector用于存放点对，对两幅图中的BRIEF描述子进行匹配，使用Hamming距离
//6.筛选匹配点对（距离小于最小值的两倍）
//7.绘制匹配结果
vector< vector<Point2f> > orb_match(Mat img1,Mat img2,string outPath)
{
  //Step1

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);

  //  vector<Point2f> kpoints1, kpoints2;
  //  for(vector<cv::KeyPoint>::iterator vit=keyPoint1.begin(); vit!=keyPoint1.end();vit++)
  //  {
  //     cout << (*vit).pt.x << " " << (*vit).pt.y << endl;
  //     kpoints1.push_back((*vit).pt);
  //  }
  //  //Mat img1_;//(img1.size(),CV_8UC3);
  //  //cvtColor(img1, img1_, COLOR_GRAY2RGB);
  // cout << "--------------"<< endl;


  // //KeyPoint::convert(keyPoint1,kpoints1,1,1,0,-1);
  // //KeyPoint::convert(keyPoint2,kpoints2,1,1,0,-1);

  // TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,40,0.01);
  // cornerSubPix(img1, kpoints1, cv::Size(5, 5), cv::Size(-1, -1), criteria);
  // //cornerSubPix(img2, kpoints2, cv::Size(5, 5), cv::Size(-1, -1), criteria);

  //  for(vector<Point2f>::iterator vit=kpoints1.begin(); vit!=kpoints1.end();vit++)
  //  {
  //     cout << (*vit).x << " " << (*vit).y << endl;

  //  }

  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  cout << "keypoints1 : " << keyPoint1.size() << "  keypoints2: " << keyPoint2.size() << endl;
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 150;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<Point2f> points1;
  vector<Point2f> points2;
  vector<DMatch> good_matches;

  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
      //注意这两个Idx是不一样的
      points1.push_back(keyPoint1[matches[i].queryIdx].pt);
      points2.push_back(keyPoint2[matches[i].trainIdx].pt);
      octave1.push_back(keyPoint1[matches[i].queryIdx].octave);
      octave2.push_back(keyPoint2[matches[i].trainIdx].octave);
    }
  }

  int ransacReprojThreshold = 5;
  Mat H12;
  H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
  matchesMask.resize(good_matches.size(), 0 );  
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for( size_t i1 = 0; i1 < points1.size(); i1++ )  //保存‘内点’
	{
		if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记
		{
			matchesMask[i1] = 1;
		}	
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches,match_img2,Scalar(0,0,255),Scalar::all(-1),matchesMask);
  	imwrite(outPath,match_img2);


  vector< vector<Point2f> > result;
  result.push_back(points1);
  result.push_back(points2);
  
  //Step7
  //Mat outImg;
  //drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches,outImg);
  //imwrite(outPath,outImg);
  cout<<"保留的匹配点对："<<good_matches.size()<<endl;
  
  return result;
}

float computerConstrast(const Mat img);

int main(int argc, char** argv)
{

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
  //-- 读取图像
  Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
  Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
  //string img1 = "left.PNG";
  //string img2 = "right.PNG";
  string outpath = "matching.png";
  vector< vector<Point2f> > points;
  vector<Point2f> points1;
  vector<Point2f> points2;
  points = orb_match(img_1,img_2,outpath);
  points1 = points[0];
  points2 = points[1];
  
  int num = 0;
  for(int i=0;i<points1.size();i++)
  {
    if(matchesMask[i] == 0)
      continue;
    cout<<"第"<<(i+1)<<"对点：";
    cout<<"("<<points1[i].x<<","<<points1[i].y<<") " << octave1[i];
    cout<<" - ";
    cout<<"("<<points2[i].x<<","<<points2[i].y<<") " << octave2[i];
    
    if(points1[i].x == points2[i].x && points1[i].y == points2[i].y)
	  {
      num ++;
    }
    else
    {
      cout << "----different" << endl;
    }
    
    cout<<endl;
  }

computerConstrast(img_1);
computerConstrast(img_2);
cout << "---------------"<<endl;  
cout << "total matches: " << points1.size() << " same: " << num << " property: " << (float)num / (float)points1.size() << endl;
  return 1;
}

float computerConstrast(const Mat img){
  Scalar mean; 
  Scalar stddev; 

  cv::meanStdDev( img, mean, stddev ); 
  double mean_pxl = mean.val[0]; 
  double stddev_pxl = stddev.val[0]; 
  double sum = 0;
  double constrast;
    int M = img.cols;
    int N = img.rows;
    for(int i = 0; i < N; i ++)
    {
      for(int j = 0; j < M; j++)
      {
        sum += (img.at<uchar>(i,j) - mean_pxl)*(img.at<uchar>(i,j) - mean_pxl);
      }
    }

    constrast = sqrt(sum / M / N);

    cout << "Image Constrast: " << constrast << "  stddev_pxl: " << stddev_pxl << endl;

}
