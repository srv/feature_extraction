/*
 *  Surf.cpp
 *
 *  Created by Volker Nannen on 01-05-11.
 *  Copyright 2010 Systems, Robotics and Vision Group,
 *  Universitat de les Illes Balears.
 *  All rights reserved.
 *
 */

/***********************************************************
 *  This file contains code from OpenSURF, which is         *
 *  distributed under the GNU GPL. For more information     *
 *  on OpenSURF use the contact form at                     *
 *  http://www.chrisevansdev.com                            *
 *                                                          *
 *  The OpenSURF code is copyrighted by                     *
 *  C. Evans, Research Into Robust Visual Features,         *
 *  MSc University of Bristol, 2008.                        *
 ************************************************************/

#ifndef SURF_H
#define SURF_H


#include "opencv2/opencv.hpp"

//#define OpenSURFcpp
//#define ODOMETRY
#define OpenSURF_COMPATIBLE
//#define NEW_DESCRIPTOR
//#define SURF_MARKER

#ifdef OpenSURFcpp // defined in Surf.h
#include "OpenSURF.h"
#endif

namespace odometry {

#ifdef ODOMETRY
class KeyPoint;
#endif

//-------------------------------------------------------

class Surf {

public:

  static const int OCTAVES = 5;
  static const int INTERVALS = 4;
  static const int INIT_STEP = 2;
  static const float THRESHOLD_RESPONSE = 26; // 0.0004f for float integral
  static const int MAX_POINTS = 200;
  static const int BOX_X = 0;
  static const int BOX_Y = 0;
  static const float DESCRIPTOR_SCALE_FACTOR = 1;

  Surf( const cv::Mat & source,
        int octaves = OCTAVES,
        int init_step = INIT_STEP,
        float thres = THRESHOLD_RESPONSE,
        int max_points = MAX_POINTS,
        int boxX = BOX_X,
        int boxY = BOX_Y );

  Surf( int octaves = OCTAVES,
        int init_step = INIT_STEP,
        float thres = THRESHOLD_RESPONSE,
        int max_points = MAX_POINTS,
        int boxX = BOX_X,
        int boxY = BOX_Y );

  //! Destructor
  ~Surf();

  void init( int octaves = OCTAVES,
             int init_step = INIT_STEP,
             float thres = THRESHOLD_RESPONSE,
             int max_points = MAX_POINTS,
             int boxX = BOX_X,
             int boxY = BOX_Y );

  void init( const cv::Mat & source );
  void init( const cv::Mat & source,
             const cv::Mat & mask );

  int size() { return surfPoints.size(); }

  void detect();

  void detect( std::vector<cv::KeyPoint> & keys, 
               bool upright = false );
  
  void detect( const cv::Mat & source,
               std::vector<cv::KeyPoint> & keys,
               const cv::Mat & mask = cv::Mat(), 
               bool upright = false );


#ifdef ODOMETRY
  void compute( KeyPoint * points,
                bool upright = false );
#endif
  
  void compute( std::vector<cv::KeyPoint> & keys,
               cv::Mat & descriptors );
  
  void compute( const cv::Mat & source,
                std::vector<cv::KeyPoint> & keys,
                cv::Mat & descriptors );

  void operator()( const cv::Mat & source,
                   const cv::Mat & mask,
                   std::vector<cv::KeyPoint> & keys,
                   cv::Mat & descriptors,
                   bool upright = false );

#ifdef OpenSURFcpp
  OpenSURF::Timer timer;
#endif

private:

  struct Layer {
    Layer( int width,
           int height,
           int step,
           int filter,
           int m )
    : width( width / m ), height( height / m ), step( step * m ),
//    : width( width ), height( height ), step( step ),
//    : width( width * 2 ), height( height * 2 ), step( 1 ),
      filter( 3 * ( filter * m + 1 ) ) {
//        std::cout << " filter size " << 0.133f * this->filter << std::endl;
      assert( width > 0 and height > 0 );

      responses = new float[this->width * this->height];
      memset( responses, 0, sizeof( float ) * this->width * this->height );

      laplacian = new float[this->width * this->height];
      memset( laplacian, 0, sizeof( float ) * this->width * this->height );
    }

    ~Layer() {
      if ( responses ) delete [] responses;
      if ( laplacian ) delete [] laplacian;
    }

    int width, height, step, filter;
    float * responses;
    float * laplacian;

  };

  struct Extremum {
    Extremum() {; }
    Extremum( float response,
              float laplacian,
              int x,
              int y,
              int box,
              unsigned char b,
              unsigned char m,
              unsigned char t ) :
      response( response ), laplacian( laplacian ), x( x ), y( y ),
      box( box ), b( b ), m( m ), t( t )
    {; }
    bool operator<( const Extremum & other ) const {
      return response > other.response;
    }
    float response;
    float laplacian;
    int x, y;
    int box;
    unsigned char b, m, t;
  };

  struct SurfPoint {
    float x, y, scale;
    //! Orientation measured anti-clockwise from +ve x-axis
    float response;
    float laplacian;
  };

  struct Angle {
    float a, x, y;
    bool operator<( const Angle & other ) const { return a < other.a; }
  };

  void clear();

  template<class T> void integrate( const cv::Mat & source );

  void buildPyramide();
  template<class T> void buildLayer( Layer * r );

  void getExtrema();
  void interpolate( Extremum & );
  void takeBest();

  //! Assign the current Ipoint an orientation
  template<class T> float getOrientation( SurfPoint & surfPoint );

  //! Get the descriptor. See Agrawal ECCV 08
  template<class T> void getDescriptor( float keyPointX,
                                        float keyPointY,
                                        float keyPointScale,
                                        float orientation,
                                        float * descriptor );

  template<class T> void getDescriptorUpright( float keyPointX,
                                               float keyPointY,
                                               float keyPointScale,
                                               float * descriptor );

  //! Get the angle from the +ve x-axis of the vector given by (X Y)
  inline float getAngle( float x,
                         float y ) {
// first option for interval [0, 2*pi), second for interval (-pi, pi]
//    return ( x ? atan( y / x ) : ( y >= 0 ? 3.14159f / 2 : -3.14159f / 2 ) )
//           + ( x >= 0 ? ( y >= 0 ? 0 : 2 * 3.14159f ) : 3.14159f );
//    return ( x ? atan( y / x ) : ( y >= 0 ? 3.14159f / 2 : -3.14159f / 2 ) )
//           + ( x >= 0 ? 0 : ( y >= 0 ? 3.14159 : -3.14159f ) );
    return atan2( y, x );
  }

  inline int fastFloor( const float x ) {
//#ifdef  OpenSURF_COMPATIBLE
    return x >= 0 ? (int)x : ( (int)x == x ? (int)x : ( (int)x ) - 1 );
//#else
//    return x >= 0 ? x : x - 1;
//#endif
  }

  bool isInt() { return ( imgRows * imgCols ) <= ( 1 << 23 ); }

  inline int fastMin( int a,
                      int b ) { return ( a < b ) ? a : b; }
  inline int fastMax( int a,
                      int b ) { return ( a > b ) ? a : b; }

  //---------------- Private Variables -----------------//

  //! Pointer to the integral Image, and its attributes
  void * integral;
  int imgRows, imgCols;


  //! Reference to vector of features passed from outside
  std::vector<SurfPoint> surfPoints;
  std::vector<Extremum> extrema;

  //! Response stack of determinant of hessian values
  Layer * * pyramide;
  int pyramideSize;

  int octaves;
//  int intervals;
  int initStep;
  int maxPoints;
  float thresholdResponse;
  bool upright;
  int boxX, boxY;

  static const int filter_map[OCTAVES][INTERVALS];
  static float gauss_s1[109];
  static float gauss_s2[16];
  static const char * className;
};

}

#endif
