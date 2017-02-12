#include <jni.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
const int DEFUALT_MARKER = 0;
const int DEFAULT_MARKER_EDGE = 128;
const int DEFAULT_MARKER_WALL = 255;

const int DEV = 0;
const int DEV2 = 0;






/**
 * Function to initialize markers in the image.
 * It takes
 * 		src          input image ( 3-channel RGB or grayscale )
 * Returns the markers ( image of the same size with labels as pixel values ).
 */
Mat initMarkers(Mat src) {
    Mat gray, sobelX, sobelY;
    if(src.type() == CV_8UC3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    Sobel(gray, sobelX, -1, 1, 0);
    Sobel(gray, sobelY, -1, 0, 1);
    Mat sobelNet = abs(sobelX) + abs(sobelY);
    Scalar mean, stdDev;
    meanStdDev(sobelNet, mean, stdDev);
    int s = 5;
    double t = -3*stdDev[0];

    Mat thresh;
    adaptiveThreshold(sobelNet, thresh, DEFAULT_MARKER_EDGE, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, s, t); //EDIT from GAUSSIAN THRESH
    //if(DEV2) imshow("thresh", thresh);
    // waitKey(0);
    Mat dst;
    thresh.convertTo(dst, CV_32S);
    // cout << dst.type() << endl << CV_32SC1 << endl;
    return dst;
}

/**
 * Utility function to extract wall's mask.
 */
void extractMask(Mat src, int pts[][2], int n_pts, Mat markersOrig, Mat & mask) {
    int x, y;
    Mat markers;
    markersOrig.copyTo(markers);
    for(int i=0; i<n_pts; ++i) {
        x = pts[i][0], y = pts[i][1];
        markers.at<int>(y, x) = DEFAULT_MARKER_WALL;
    }

    // cout << 61 << " " << src.type() << " " << CV_8UC3 << endl;
    watershed(src, markers);

    Mat _markers;
    markers.convertTo(_markers, CV_8U);

    //if(DEV) imshow("markers", _markers);

    threshold(_markers, mask, DEFAULT_MARKER_WALL - 1, 255, THRESH_BINARY);
}

/**
 * Utility function to match illuminations in src and target image.
 * Uses L of LAB space as a measure of illumination and makes the L channel same
 * for both the images.
 */
void matchIllumination(Mat src, Mat & target, Mat mask) {

    Mat srcLab, targetLab, targetLabMatched;
    cvtColor(src, srcLab, COLOR_BGR2YUV);
    cvtColor(target, targetLab, COLOR_BGR2YUV);

    vector<Mat> srcLabSplits;
    split(srcLab, srcLabSplits);

    vector<Mat> targetLabSplits;
    split(targetLab, targetLabSplits);

    vector<Mat> targetLabSplitsMatched;
    targetLabSplitsMatched.push_back(srcLabSplits[0]);
    targetLabSplitsMatched.push_back(targetLabSplits[1]);
    targetLabSplitsMatched.push_back(targetLabSplits[2]);

    merge(targetLabSplitsMatched, targetLabMatched);
    cvtColor(targetLabMatched, target, COLOR_YUV2BGR);

}

/**
 * Utility function to fill the wall, given the source image, mask and the wall.
 * Generic base function for both color/pattern based painting.
 */
void wallFill(Mat src, Mat wall, Mat mask, Mat & dst) {
    Mat maskInv = Mat(mask.size(), CV_8UC1, Scalar(255, 255, 255)) - mask;
    Mat m1, m2, m3;

    //if(DEV) imshow("mask", mask);

    matchIllumination(src, wall, mask);

    bitwise_and(src, src, m1, maskInv);
    bitwise_and(wall, wall, m2, mask);

    m3 = m1 + m2;
//bilateralFilter(m3, dst, 3, 0.5, 0.5);

    GaussianBlur(m3,dst,Size(3,3),0.5);
}

/**
 * Function to mark a particular region as wall ( truth ), and extends the
 * wall-group throughout the image. It takes
 * 		src          input image ( 3-channel RGB )
 * 		pts          1D array of points ( mouse-clicks or touches )
 * 		n_pts        length of pts array
 * 		markersOrig  markers ( to be obtained from initMarkers function )
 * 		wall         image depicting a wall
 * Returns the wall-painted image.
 */
Mat wallFillPattern(Mat src, int pts[][2], int n_pts, Mat markersOrig, Mat wall) {
    Mat mask, maskInv, dst;
    extractMask(src, pts, n_pts, markersOrig, mask);

    wallFill(src, wall, mask, dst);
    return dst;
}

/**
 * Function to mark a particular region as wall ( truth ), and extends the
 * wall-group throughout the image. It takes
 * 		src          input image ( 3-channel RGB )
 * 		pts          1D array of points ( mouse-clicks or touches )
 * 		n_pts        length of pts array
 * 		markersOrig  markers ( to be obtained from initMarkers function )
 * 		color        scalar denoting the color
 * Returns the wall-painted image.
 */
Mat wallFillColor(Mat src, int pts[][2], int n_pts, Mat markersOrig, Scalar color) {
    Mat mask, maskInv, dst;
    extractMask(src, pts, n_pts, markersOrig, mask);

    Mat wall = Mat(mask.size(), CV_8UC3, color);
    wallFill(src, wall, mask, dst);
    return dst;
}

/** UTIL FUNCTIONS FOR SAMPLE RUN ON A COMPUTER == START **/
vector<Point> pts;
Point prevPt(-1, -1), INVALID_POINT(-1, -1);
char * out_name;
Mat _img, _markers, _wall;
Scalar _color;
/*
Rescales the image to 1080 width, and centers the image, padding to 1920x1080 in total
satisfying image.cols<width (1080 default) , image.rows<height (1920 default), such that

 */

Mat rescale(Mat src ,int fill_border=0, int width=1080,int height=1920 )
{
    float aspect_ratio = src.cols/((float)(src.rows));
    Scalar fill_color;
    if (fill_border==0)
        fill_color = Scalar(0,0,0); //Black fill_color
    else
        fill_color = Scalar(255,255,255);

    Size dst_Size = Size(1920,1080);
    Mat dst;
    int top,bottom,left,right;
    if (aspect_ratio>16.0/9.0)
    {
        left = 0;
        right = 0;
        top  = (int) (9*src.cols-16*src.rows)/(32.0);
        bottom = top;
    }
    else
    {
        top = 0 ;
        bottom = top;
        left = (int) (16*src.rows-9*src.cols)/(18.0);
        right = left;
    }
    copyMakeBorder(src,src,top,bottom,left,right,BORDER_CONSTANT,fill_color);
    resize(src,dst,dst_Size,0,0,INTER_AREA);
    return dst;
}


extern "C"
{
void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                           jlong matAddrGray,jlong optaddr,
                                                                           jint nbrElem) {
    Mat &mGr = *(Mat *) matAddrGray;
    cvtColor(mGr,mGr,COLOR_RGBA2RGB);
    Mat mgr;
    resize(mGr,mgr,Size(0,0),0.25,0.25);
    Scalar color(255,0,0);
    int pts_arr1[1][2];
    Mat markersOrig = initMarkers(mgr);
    pts_arr1[0][0] = (int) mgr.rows/2.0;
    pts_arr1[0][1] = (int) mgr.cols/2.0;
    Mat &opt = *(Mat *) optaddr;
    opt = wallFillColor(mgr,pts_arr1,1,markersOrig,color);
    resize(opt,opt,mGr.size());
    //opt = mGr;


}
}