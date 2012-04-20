/*
 *  smart_surf.cpp
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

#ifndef VISUAL_ODOMETRY_SMART_SURF
#define VISUAL_ODOMETRY_SMART_SURF

//#define VISUAL_ODOMETRY
#define OPEN_SURF_COMPATIBLE
//#define NEW_DESCRIPTOR
//#define PYRAMIDE // to build response layers as pyramide
//#define TIMER

#include "opencv2/opencv.hpp"


#ifdef VISUAL_ODOMETRY
#include "odometry/utilities.h"
#include "odometry/open_surf.h"
#endif

namespace visual_odometry {

class KeyPoint;

//-------------------------------------------------------

class SmartSurf {

public:

  static const int OCTAVES       = 5;
  static const int INTERVALS     = 4;
  static const float THRESHOLD   = 26; // 0.0004f for float integral
  static const int INIT_STEP     = 2;
  static const bool USE_PYRAMIDE = true;
  static const bool INTERPOLATE  = true;

  SmartSurf( const cv::Mat & source,
        int octaves = OCTAVES,
        float thres = THRESHOLD,
        int init_step = INIT_STEP,
        bool use_pyramide = USE_PYRAMIDE,
        bool interpolate = INTERPOLATE );

  SmartSurf( int octaves = OCTAVES,
        float thres = THRESHOLD,
        int init_step = INIT_STEP,
        bool use_pyramide = USE_PYRAMIDE,
        bool interpolate = INTERPOLATE  );

  //! Destructor
  ~SmartSurf();

  void init( int octaves = OCTAVES,
             float thres = THRESHOLD,
             int init_step = INIT_STEP,
             bool use_pyramide = USE_PYRAMIDE,
             bool interpolate = INTERPOLATE  );

  void init( const cv::Mat & source );
  void init( const cv::Mat & source,
             const cv::Mat & mask );

  int size() const { return surf_points_.size(); }

  void detect( visual_odometry::KeyPoint * & points,
               int & length );

  void detect( std::vector<cv::KeyPoint> & keys,
               bool upright = false );

  void detect( const cv::Mat & source,
               std::vector<cv::KeyPoint> & keys,
               const cv::Mat & mask = cv::Mat(),
               bool upright = false );


  void compute( visual_odometry::KeyPoint * points,
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

#ifdef TIMER
  Timer timer;
#endif

private:

  struct Layer {
    Layer( int step,
           int filter,
           int cols,
           int rows );
    ~Layer() {
      delete[] hessian;
      delete[] laplacian;
    }
    float variance() const { return 1.44f * ( filter * filter ) / 81.0f; }
    int step, filter, cols, rows;
    float * hessian;
    float * laplacian;
  };

  struct Extremum {
    Extremum() {; }
    Extremum( int x,
              int y,
              short octave,
              short interval ) :
      x( x ), y( y ), octave( octave ), interval( interval )  {; }
    int x, y;
    short octave, interval;
  };

  struct SurfPoint {
    float x, y, scale;
    //! Orientation measured anti-clockwise from +ve x-axis
    float hessian;
    float laplacian;
    unsigned char octave;
  };

  struct Angle {
    float a, x, y;
    bool operator<( const Angle & other ) const { return a < other.a; }
  };

  void clear();

  template<class T> void integrate( const cv::Mat & source );

  template<class T> void getResponse( Layer * r );

  void getExtrema();
  void interpolateExtremum( Extremum & );

  //! Assign the current Ipoint an orientation
  template<class T> float getOrientation( float key_point_x,
                                          float key_point_y,
                                          float key_point_scale );

  //! Get the descriptor. See Agrawal ECCV 08
  template<class T> void getDescriptor( float key_point_x,
                                        float key_point_y,
                                        float key_point_scale,
                                        float orientation,
                                        float * descriptor );

  template<class T> void getDescriptorUpright( float key_point_x,
                                               float key_point_y,
                                               float key_point_scale,
                                               float * descriptor );

  inline int fastFloor( const float x ) {
//#ifdef  OPEN_SURF_COMPATIBLE
    return x >= 0 ? (int)x : ( (int)x == x ? (int)x : ( (int)x ) - 1 );
//#else
//    return x >= 0 ? x : x - 1;
//#endif
  }

  void detect();

  bool isInt() { return ( img_rows_ * img_cols_ ) <= ( 1 << 23 ); }

  inline int fastMin( int a,
                      int b ) { return ( a < b ) ? a : b; }
  inline int fastMax( int a,
                      int b ) { return ( a > b ) ? a : b; }

  void * integral_;
  int img_rows_, img_cols_;

  std::vector<SurfPoint> surf_points_;
  std::vector<Extremum> extrema_;

  Layer * * pyramide_;
  int pyramide_size_;

  int octaves_;
  int intervals_;
  float threshold_;
  int init_step_;
  bool use_pyramide_;
  bool interpolate_;

  int max_points_;
  bool upright_;
  int box_x, box_y_;


  static const int FILTER_MAP[OCTAVES][INTERVALS];
  static float gauss_s1[109];
  static float gauss_s2[16];

public:
  static const char * CLASS_NAME;
  friend std::ostream & operator<<( std::ostream &,
                                    const SmartSurf & );
};

}

#endif
