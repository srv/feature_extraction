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
//#define PYRAMIDE // to build response layers as pyramide

#ifdef OpenSURFcpp // defined in Surf.h
#include "OpenSURF.h"
#endif

namespace odometry {

class KeyPoint;

//-------------------------------------------------------

class Surf {

public:

  static const int OCTAVES = 5;
  static const int INTERVALS = 4;
  static const int INIT_STEP = 2;
  static const float THRESHOLD_RESPONSE = 26; // 0.0004f for float integral
  static const float DESCRIPTOR_SCALE_FACTOR = 1;

  Surf( const cv::Mat & source,
        int octaves = OCTAVES,
        int init_step = INIT_STEP,
        float thres = THRESHOLD_RESPONSE );

  Surf( int octaves = OCTAVES,
        int init_step = INIT_STEP,
        float thres = THRESHOLD_RESPONSE );

  //! Destructor
  ~Surf();

  void init( int octaves = OCTAVES,
             int init_step = INIT_STEP,
             float thres = THRESHOLD_RESPONSE );

  void init( const cv::Mat & source );
  void init( const cv::Mat & source,
             const cv::Mat & mask );

  int size() { return surfPoints.size(); }

  void detect( odometry::KeyPoint * & points,
               int & length );

  void detect( std::vector<cv::KeyPoint> & keys, 
               bool upright = false );
  
  void detect( const cv::Mat & source,
               std::vector<cv::KeyPoint> & keys,
               const cv::Mat & mask = cv::Mat(), 
               bool upright = false );
 

  void compute( odometry::KeyPoint * points,
                int length,
                bool upright = false );
  
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
#ifdef PYRAMIDE
    : width( width / m ), height( height / m ), step( step * m ),
#else
    : width( width ), height( height ), step( step ),
    //    : width( width * 2 ), height( height * 2 ), step( 1 ),
#endif
      filter( 3 * ( filter * m + 1 ) ) {
//        std::cout << " filter size " << 0.133f * this->filter << std::endl;

      responses = new float[this->width * this->height];
      memset( responses, 0, sizeof( float ) * this->width * this->height );

      laplacian = new float[this->width * this->height];
      memset( laplacian, 0, sizeof( float ) * this->width * this->height );
    }

    ~Layer() {
      delete[] responses;
      delete[] laplacian;
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
              short octave,
              short interval ) :
      response( response ), laplacian( laplacian ), x( x ), y( y ),
      octave( octave ), interval( interval )  {; }
    float response, laplacian;
    int x, y;
    int octave, interval;
  };

  struct SurfPoint {
    float x, y, scale;
    //! Orientation measured anti-clockwise from +ve x-axis
    float response;
    float laplacian;
    unsigned char octave;
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

  //! Assign the current Ipoint an orientation
  template<class T> float getOrientation( float keyPointX,
                                          float keyPointY,
                                          float keyPointScale );

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

  inline int fastFloor( const float x ) {
//#ifdef  OpenSURF_COMPATIBLE
    return x >= 0 ? (int)x : ( (int)x == x ? (int)x : ( (int)x ) - 1 );
//#else
//    return x >= 0 ? x : x - 1;
//#endif
  }

  void detect();
  
  bool isInt() { return ( imgRows * imgCols ) <= ( 1 << 23 ); }

  inline int fastMin( int a,
                      int b ) { return ( a < b ) ? a : b; }
  inline int fastMax( int a,
                      int b ) { return ( a > b ) ? a : b; }

  void * integral;
  int imgRows, imgCols;

  std::vector<SurfPoint> surfPoints;
  std::vector<Extremum> extrema;

  Layer * * pyramide;
  int pyramideSize;

  int octaves;
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
