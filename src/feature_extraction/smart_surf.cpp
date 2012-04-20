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

#include "opencv2/core/core.hpp"

#include "feature_extraction/smart_surf.h"

#ifdef VISUAL_ODOMETRY
#include "odometry/key_point.h"
#include "odometry/statistics.h"
#include "odometry/debug.h"
#else   // VISUAL_ODOMETRY
#define tellu( ... )
#endif  // VISUAL_ODOMETRY

namespace visual_odometry {

//***************************************************************
//*** SmartSurf
//***************************************************************

const char * SmartSurf::CLASS_NAME = "SmartSurf";

const int SmartSurf::OCTAVES;
const int SmartSurf::INTERVALS;
const float SmartSurf::THRESHOLD;
const int SmartSurf::INIT_STEP;
const bool SmartSurf::USE_PYRAMIDE;
const bool SmartSurf::INTERPOLATE;

const int SmartSurf::FILTER_MAP[OCTAVES][INTERVALS] = {{0, 1, 2, 3},
                                                  {1, 3, 4, 5},
                                                  {3, 5, 6, 7},
                                                  {5, 7, 8, 9},
                                                  {7, 9, 10, 11}};
#define pi 3.14159f
#define pi2 ( 3.14159f * 2 )
float SmartSurf::gauss_s1[109] = {0};
float SmartSurf::gauss_s2[16] = {0};

//-------------------------------------------------------

SmartSurf::SmartSurf( const cv::Mat & source,
            int octaves,
            float threshold,
            int init_step,
            bool use_pyramide,
            bool interpolate ) :
  integral_( 0 ),
  img_rows_( 0 ),
  img_cols_( 0 ),
  pyramide_( 0 ),
  pyramide_size_( 0 ) {
  init( source );
  init( octaves, threshold, init_step, use_pyramide, interpolate );
  //  std::cout << "img_rows_ " << img_rows_
  //  << ", img_cols_ " << img_cols_ << std::endl;
}

SmartSurf::SmartSurf( int octaves,
            float threshold,
            int init_step,
            bool use_pyramide,
            bool interpolate  ) :
  integral_( 0 ),
  img_rows_( 0 ),
  img_cols_( 0 ),
  pyramide_( 0 ),
  pyramide_size_( 0 ) {
  init( octaves, threshold, init_step, use_pyramide, interpolate );
}

void SmartSurf::init( int octaves,
                 float threshold,
                 int init_step,
                 bool use_pyramide,
                 bool interpolate  ) {
  tellu( "" );
  this->octaves_      = octaves > 0 and octaves < OCTAVES ? octaves : OCTAVES;
  this->intervals_    = 2;
  this->threshold_    = threshold >= 0 ? threshold : THRESHOLD;
  this->init_step_    = init_step > 0 and init_step <= 6 ? 
                        init_step : INIT_STEP;
  this->use_pyramide_ = use_pyramide;
  this->interpolate_  = interpolate;
//  float y1 = -10, y2 = 0, y3 = 10;
//  for ( float x = -10; x < 20; x++ ) {
//    printf( "% 2.3f, % 2.3f, % 2.3f :  % 2.3f, % 2.3f, % 2.3f\n",
//            atan2( y1, x ), atan2( y2, x ), atan2( y3, x ),
//            atan2( x, y1 ), atan2( x, y2 ), atan2( x, y3 ) );
//  }
//  exit( 0 );

  if ( !gauss_s2[0] ) {
    float gauss25[6][6] = {
      {0.02350693969273, 0.01849121369071, 0.01239503121241,
       0.00708015417522, 0.00344628101733, 0.00142945847484},
      {0.02169964028389, 0.01706954162243, 0.01144205592615,
       0.00653580605408, 0.00318131834134, 0.00131955648461},
      {0.01706954162243, 0.01342737701584, 0.00900063997939,
       0.00514124713667, 0.00250251364222, 0.00103799989504},
      {0.01144205592615, 0.00900063997939, 0.00603330940534,
       0.00344628101733, 0.00167748505986, 0.00069579213743},
      {0.00653580605408, 0.00514124713667, 0.00344628101733,
       0.00196854695367, 0.00095819467066, 0.00039744277546},
      {0.00318131834134, 0.00250251364222, 0.00167748505986,
       0.00095819467066, 0.00046640341759, 0.00019345616757}
    };
    int k = 0;
    for ( int i = -5; i <= 5; i++ ) {
      int e = fastMin( 5, 8 + ( i > 0 ? -i : i ) );
      for ( int j = -e; j <= e; j++ ) {
        gauss_s1[k] = gauss25[ i > 0 ? i : -i ][ j > 0 ? j : -j];
        k++;
      }
    }
    k = 0;
    for ( float cx = -1.5f; cx < 2.f; cx++ ) {
      for ( float cy = -1.5f; cy < 2.f; cy++ ) {
        gauss_s2[k++] = ( 1.0f / ( pi * 2.0f * 1.5f * 1.5f ) ) *
                        exp( -( cx * cx + cy * cy ) / ( 2.0f * 1.5f * 1.5f ) );
      }
    }

  }

}

void SmartSurf::init( const cv::Mat & source ) {
  tellu( "" );
  if ( img_cols_ != source.cols or img_rows_ != source.rows ) {
    clear();
    img_cols_ = source.cols;
    img_rows_ = source.rows;
    if ( isInt() ) {
      integral_ = new int[img_rows_ * img_cols_];
    } else {
      integral_ = new int64_t[img_rows_ * img_cols_];
    }
  }
  if ( isInt() ) {
    integrate<int>( source );
  } else {
    integrate<int64_t>( source );
  }
}

void SmartSurf::init( const cv::Mat & source,
                 const cv::Mat & ) {
  tellu( "" );
  init( source );
}

//-------------------------------------------------------

template<class T> void SmartSurf::integrate( const cv::Mat & source ) {
  tellu( "" );
  T * data = (T *)integral_;
  unsigned char * img_data;
  if ( source.channels() == 1 ) {
    assert( source.isContinuous() );
    img_data = source.data;
  } else {
    img_data = new unsigned char[img_rows_ * img_cols_];
    cv::Mat tmp( source.size(), CV_8UC1, img_data );
    cvtColor( source, tmp, CV_BGR2GRAY );
  }
  T sum = 0;
  int index = 0;
  int index2 = 0;
  for (; index < img_cols_; index++ ) {
    sum += img_data[index];
    data[index] = sum;
  }
  for ( int i = 1; i < img_rows_; i++ ) {
    sum = 0;
    for ( int j = 0; j < img_cols_; j++, index++, index2++ ) {
      sum += img_data[index];
      data[index] = data[index2] + sum;
    }
  }
  if ( source.channels() != 1 ) delete[] img_data;
}

//-------------------------------------------------------

SmartSurf::~SmartSurf() {
  tellu( "" );
  clear();
}

void SmartSurf::clear() {
  tellu( "" );
  if ( pyramide_ ) {
    for ( int i = 0; i < pyramide_size_; i++ ) delete pyramide_[i];
    delete[] pyramide_;
    pyramide_ = 0;
    pyramide_size_ = 0;
  }
  if ( integral_ ) {
    if ( isInt() ) {
      delete[] (int *)integral_;
    } else {
      delete[] (int64_t *)integral_;
    }
    integral_ = 0;
    img_rows_ = 0;
    img_cols_ = 0;
  }
}

//-------------------------------------------------------

void SmartSurf::operator()( const cv::Mat & source,
                       const cv::Mat & mask,
                       std::vector<cv::KeyPoint> & keys,
                       cv::Mat & descriptors,
                       bool upright ) {
  tellu( "" );
  init( source, mask );
  detect( keys, upright );
  compute( keys, descriptors );
}
//-------------------------------------------------------

void SmartSurf::detect( visual_odometry::KeyPoint * & points,
                   int & length ) {
#ifdef VISUAL_ODOMETRY
  tellu( "KeyPoint" );
  detect();
  length = surf_points_.size();
  points = length ? new KeyPoint[length] : 0;
  for ( int i = 0; i < length; i++ ) {
    SurfPoint & surf_point = surf_points_[i];
    //  std::cout << "keypoint size " << sizeof(KeyPoint) << std::endl;
    //  std::cout << "keypointheader size "
    //  << sizeof(KeyPointHeader) << std::endl;
    points[i].set( surf_point.x,
                   surf_point.y,
                   surf_point.scale,
                   0,
                   surf_point.hessian,
                   surf_point.laplacian > 0 ? 1 : -1 );
  }
#endif  // VISUAL_ODOMETRY
}

void SmartSurf::detect( const cv::Mat & source,
                   std::vector<cv::KeyPoint> & keys,
                   const cv::Mat & mask,
                   bool upright ) {
  tellu( "cv::Mat" );
  init( source, mask );
  detect( keys, upright );
}

void SmartSurf::detect( std::vector<cv::KeyPoint> & keys,
                   bool upright ) {
  tellu( "std::vector<cv::KeyPoint>" );
  detect();
  keys.clear();
  keys.resize( surf_points_.size() );
  for ( unsigned int i = 0; i < surf_points_.size(); i++ ) {
    keys[i].pt.x = surf_points_[i].x;
    keys[i].pt.y = surf_points_[i].y;
    keys[i].size = surf_points_[i].scale;
    keys[i].angle = upright ? ( isInt() ?
                                getOrientation<int>( surf_points_[i].x,
                                                     surf_points_[i].y,
                                                     surf_points_[i].scale ) :
                                getOrientation<int64_t>( surf_points_[i].x,
                                                         surf_points_[i].y,
                                                         surf_points_[i].scale )
                                * ( 360.f / pi2 ) ) : 0;
    keys[i].angle += keys[i].angle < 0 ? 360.f : 0;
    keys[i].response = surf_points_[i].hessian;
    keys[i].octave = surf_points_[i].octave;
  }
}

void SmartSurf::detect() {
  tellu( "" );
#ifdef TIMER
  double time = cv::getTickCount();
#endif
  // Calculate response for the first 4 octaves:
  // Oct1: 9,  15, 21, 27
  // Oct2: 15, 27, 39, 51
  // Oct3: 27, 51, 75, 99
  // Oct4: 51, 99, 147,195
  // Oct5: 99, 195,291,387

  // Bay, Ess, Tuytelaar, van Gool - Speeded-Up Robust Features (SURF) - 2008
  // page 349, left column:
  // The 9-by-9 box filters are approximations of a Gaussian with sigma=1.2
  // and represent the lowest scale (i.e. highest spatial resolution) for
  // computing the blob response maps.
  // page 349, right column:
  // The output of the 9-by-9 filter is considered as the initial scale layer
  // to which we will refer as scale sigma=1.2 (approximating Gaussian
  // derivatives with sigma=1.2) .
  int factor = 1;
  int step = init_step_;
  int cols = img_cols_ / step;
  int rows = img_rows_ / step;
  if ( !pyramide_ ) {
    pyramide_size_ = ( octaves_ + 1 ) * 2;
    pyramide_ = new Layer * [ pyramide_size_ ];
    pyramide_[0] = new Layer( step, 6 * 1 * factor + 3, cols, rows );
    pyramide_[1] = new Layer( step, 6 * 2 * factor + 3, cols, rows );
    for ( int i = 2; i <= octaves_ * 2; i += 2 ) {
      pyramide_[i]     = new Layer( step, 6 * 3 * factor + 3, cols, rows );
      pyramide_[i + 1] = new Layer( step, 6 * 4 * factor + 3, cols, rows );
      factor *= 2;
      if ( use_pyramide_ ) {
        step *= 2;
        cols = img_cols_ / step;
        rows = img_rows_ / step;
      }
    }
  }
//  for ( int i=0; i<pyramide_size_; i++ ) {
//    float scale = sqrt(pyramide_[i]->variance());
//    printf("%i & %i & %3.2f & %3.2f \\\\\n", i+1, pyramide_[i]->filter,
//           scale, i ? scale / sqrt(pyramide_[i-1]->variance()) : 0 );
//  }
#ifdef TIMER
  timer.a = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  if ( isInt() ) {
    for ( int i = 0; i < pyramide_size_; i++ ) getResponse<int>( pyramide_[i] );
  } else {
    for ( int i = 0; i < pyramide_size_; i++ ) getResponse<int64_t>( pyramide_[i] );
  }
#ifdef TIMER
  timer.b = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  extrema_.clear();
  getExtrema();
#ifdef TIMER
  timer.c = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  surf_points_.clear();
  surf_points_.reserve( extrema_.size() );
  for ( unsigned int i = 0; i < extrema_.size(); i++ ) {
    interpolateExtremum( extrema_[i] );
  }
#ifdef TIMER
  timer.d = cv::getTickCount() - time;
#endif
}

//-------------------------------------------------------

//! Calculate DoH response for supplied layer
template<class T> void SmartSurf::getResponse( Layer * layer ) {
  tellu( "" );
//  int m = 0;
  T * data = (T *)integral_;
  float * hessian = layer->hessian;             // hessian storage
  float * laplacian = layer->laplacian;         // laplacian storage
                                                //  int l = layer->filter;
                                                // filter size,  3*(2*n+1)
  int l2 = layer->filter / 2;                   // filer border,    3*n+1
  int l3 = layer->filter / 3;                   // filter lobe,     2*n+1
  int l6 = layer->filter / 6;                   //                    n
                                                // normalisation factor
  float inverse_area = 1.f / ( layer->filter * layer->filter );
  int img_cols_step = img_cols_ * layer->step;
  int l2Cols = l2 * img_cols_;
  int l3Cols = l3 * img_cols_;
  int l6Cols = l6 * img_cols_;
  int height = img_cols_ * ( img_rows_ - 1 );
  int width = img_cols_ - 1;
  int index = 0;

#ifdef SMART_HESSIAN
  int ddsize = ( layer->cols + l3 + 1 ) * ( layer->height + l3 + 1 );
  float DDxy( layer->cols + l3 + 1 ) * ( layer->height + l3 + 1 );
  for ( int ar = 0, r = -img_cols_;
        ar < layer->height + l3 + 1;
        ++ar, r += img_cols_step ) {

    for ( int ac = 0, c = -1;
          ac < layer->cols + l3 + 1;
          ++ac, c += layer->step ) {

      int r1a =  fastMin( r - l3Cols, height );
      int r2a =  fastMin( r,          height );
      int c1a =  fastMin( c - l3,     width  );
      int c2a =  fastMin( c,          width  );

      DDxy[index] = r2a < 0 or c2a < 0 ? 0 :
                    ( r1a < 0 ? 0 : ( c1a < 0 ? 0 : data[r1a + c1a] -
                                      (             data[r1a + c2a] ) ) ) -
                    (                 c1a < 0 ? 0 : data[r2a + c1a] -
                                      (             data[r2a + c2a] ) );
      index++;
    }
  }
  index = 0;
#endif

  // image coordinates: r, c
  for ( int ar = 0, r = 0;       //-img_cols_;
        ar < layer->rows;
        ++ar, r += img_cols_step ) {

    for ( int ac = 0, c = 0;       // -1;
          ac < layer->cols;
          ++ac, c += layer->step ) {
//      m++;
      int r1a =        ( r - l3Cols                   );
      int r2a = fastMin( r + l3Cols - img_cols_, height );
      int c1a =        ( c - l2 - 1                   );
      int c2a = fastMin( c + l2,               width  );
      int c1b =        ( c - l6 - 1                   );
      int c2b = fastMin( c + l6,               width  );

      float Dxx  =
        r1a < 0 ? (
          (                       data[r2a + c2a]     ) -
          (       c1a < 0 ? 0 :   data[r2a + c1a] ) -
          3 * ( (                 data[r2a + c2b]     ) -
                ( c1b < 0 ? 0 :   data[r2a + c1b] ) ) ) :
        ( (       c1a < 0 ? 0 : ( data[r1a + c1a] -
                                  data[r2a + c1a] ) ) -
          (                       data[r1a + c2a]     ) +
          (                       data[r2a + c2a]     ) -
          3 * ( ( c1b < 0 ? 0 : ( data[r1a + c1b] -
                                  data[r2a + c1b] ) ) -
                (                 data[r1a + c2b]     ) +
                (                 data[r2a + c2b]     ) ) );

      // Dyy
      r1a     =        ( r - l2Cols - img_cols_         );
      r2a     = fastMin( r + l2Cols,           height );
      int r1b =        ( r - l6Cols - img_cols_         );
      int r2b = fastMin( r + l6Cols,           height );
      c1a     =        ( c - l3                       );
      c2a     = fastMin( c + l3 - 1,           width  );

      float Dyy =
        c1a < 0 ?
        ( (                       data[r2a + c2a]     ) -
          (       r1a < 0 ? 0 :   data[r1a + c2a]  ) -
          3 * (                   data[r2b + c2a]     ) -
          (       r1b < 0 ? 0 :   data[r1b + c2a] ) ) :
        ( (       r1a < 0 ? 0 : ( data[r1a + c1a]  -
                                  data[r1a + c2a]  )  ) -
          (                       data[r2a + c1a] ) +
          (                       data[r2a + c2a] ) -
          3 * ( ( r1b < 0 ? 0 : ( data[r1b + c1a] -
                                  data[r1b + c2a] )  ) -
                (                 data[r2b + c1a] ) +
                (                 data[r2b + c2a] ) ) );

      // Dxy
      r1a     =        ( r - l3Cols - img_cols_         );
      r2a     =        ( r          - img_cols_         );
      //    r1b     = fastMin( r,                    height );
      r2b     = fastMin( r + l3Cols,           height );
      c1a     =        ( c - l3 - 1                   );
      c2a     =        ( c      - 1                   );
      //    c1b     = fastMin( c,                    width  );
      c2b     = fastMin( c + l3,               width  );

//      assert( r <= height );
//      assert( c <= width );

      bool invalid1 = c1a < 0;
      bool invalid2 = c2a < 0;
      float Dxy =
        ( r2a < 0 ? 0 :
          ( r1a < 0 ? 0 :
            ( (                  data[r1a + c  ] ) -
              (                  data[r1a + c2b] ) -
              ( invalid2 ? 0 :
                ( invalid1 ? 0 : data[r1a + c1a] ) -
                (                data[r1a + c2a] ) ) ) ) -
          ( (                    data[r2a + c  ] )  -
            (                    data[r2a + c2b] )  -
            ( invalid2 ? 0 :
              ( invalid1 ? 0 :   data[r2a + c1a] ) -
              (                  data[r2a + c2a] ) ) ) ) +
        ( invalid2 ? 0 :
          ( invalid1 ? 0 :      data[r   + c1a] -
            (                   data[r2b + c1a] ) ) -
          ( (                   data[r   + c2a] )  -
            (                   data[r2b + c2a] ) ) )  -
        (                       data[r   + c  ] ) +
        (                       data[r   + c2b] ) +
        (                       data[r2b + c  ] ) -
        (                       data[r2b + c2b] );
#ifdef SMART_HESSIAN
      int na = ( layer->cols + l3 + 1 ) *   ar            + ac + l3  + 1;
      int nb = ( layer->cols + l3 + 1 ) * ( ar + l3 + 1 ) + ac;
      int nc = ( layer->cols + l3 + 1 ) *   ar            + ac;
      int nd = ( layer->cols + l3 + 1 ) * ( ar + l3 + 1 ) + ac + l3 + 1;
      assert( DDxy[na] + DDxy[nb] - DDxy[nc] - DDxy[nd] == Dxy );
#endif

      // Normalise the filter response with respect to their size
      Dxx *= inverse_area;
      Dyy *= inverse_area;
      Dxy *= inverse_area;

      // Get the determinant of hessian and the laplacian
      hessian[index] = ( Dxx * Dyy - 0.81f * Dxy * Dxy );
      laplacian[index] = Dxx + Dyy;
      index++;
    }
  }
//  mark(m);
}

//-------------------------------------------------------

void SmartSurf::getExtrema() {
  tellu( "" );
  int total_count = 0; int total_extrema = 0; //  float total_maximum = -1e+6;
  for ( int o = 0; o < octaves_; ++o ) {
    for ( int i = 0; i < intervals_; i++ ) {
      int count = 0, extrema_count = 0; // float maximum = -1e+6;
      Layer * b = pyramide_[FILTER_MAP[o][i    ]];
      Layer * m = pyramide_[FILTER_MAP[o][i + 1]];
      Layer * t = pyramide_[FILTER_MAP[o][i + 2]];
      //      std::cout << "response layers " << o << ", " << i << ": "
      //      << b->filter - 3 << ", " << m->filter - 3 << ", "
      //      << t->filter - 3 << "; "
      //      << t->filter + b->filter - 2 * m->filter << ", "
      //      << t->filter - b->filter << std::endl;
      int border = ( t->filter + 1 ) / ( 2 * t->step );
//      mark( border, t->variance(), t->filter );
      int b_step = b->cols / t->cols;
      int b_col_step = b->cols * b_step;
      int m_step = m->cols / t->cols;
      int m_col_step = m->cols * m_step;
      int t_cols = t->cols;
      float * b_hessian = b->hessian;
      float * m_hessian = m->hessian;
      float * t_hessian = t->hessian;
      // loop over middle response layer at density of the most
      // sparse layer (always top), to find maxima across scale and space
      for ( int y = border + 1; y < t->rows - border; y++ ) {
        int m_v = y * m_col_step + border * m_step;
        int b_v = ( y - 1 ) * b_col_step + border * b_step;
        int t_v = ( y - 1 ) * t_cols;
        for ( int x = border + 1; x < t_cols - border; x++ ) {
          count++;
          m_v += m_step;
          b_v += b_step;
          float candidate = m_hessian[ m_v ];
          if ( candidate < threshold_ ) continue;
          //-----------------------------------------------
          //! Non Maximal Suppression
          // check the candidate point in the middle layer is above threshold
          // response
          int v = b_v;
          for ( int j = -1; j <= 1; j++ ) {
            if ( b_hessian[ v - b_step ] >= candidate or
                 b_hessian[ v           ] >= candidate or
                 b_hessian[ v + b_step ] >= candidate ) {
              goto no_extremum;
            }
            v += b_col_step;
          }
          v = m_v - m_col_step;
          for ( int j = -1; j <= 1; j++ ) {
            if ( m_hessian[ v - m_step ] >= candidate or
                 ( j and m_hessian[ v ] >= candidate ) or
                 m_hessian[ v + m_step ] >= candidate ) {
              goto no_extremum;
            }
            v += m_col_step;
          }
          v = t_v + x;
          for ( int j = -1; j <= 1; j++ ) {
            if ( t_hessian[ v - 1 ] >= candidate or
                 t_hessian[ v     ] >= candidate or
                 t_hessian[ v + 1 ] >= candidate ) {
              goto no_extremum;
            }
            v += t_cols;
          }
//          maximum=std::max(maximum, candidate);
          extrema_count++;
          extrema_.push_back( Extremum( /* candidate, m->laplacian[ m_v ], */
                                       x, y, o, i ) );
          //-----------------------------------------------
no_extremum:;
          //-----------------------------------------------
        }
      }
      total_extrema += extrema_count; total_count += count;
//      total_maximum = std::max(total_maximum, maximum);
//      mark( m->variance(), extrema_count, count, (float)extrema_count / count );
    }
  }
//  mark( total_extrema, total_count, (float)total_extrema / total_count );
}

//! Interpolate scale-space extrema to subpixel accuracy to form an image
// feature.

void SmartSurf::interpolateExtremum( Extremum & extremum ) {
  tellu( "" );
  Layer * b = pyramide_[FILTER_MAP[extremum.octave][extremum.interval  ]];
  Layer * m = pyramide_[FILTER_MAP[extremum.octave][extremum.interval + 1]];
  Layer * t = pyramide_[FILTER_MAP[extremum.octave][extremum.interval + 2]];
  int m_v = ( extremum.y * m->cols + extremum.x ) * m->cols / t->cols;
  if ( !interpolate_ ) {
    SurfPoint surf_point;
    surf_point.x = extremum.x * t->step;
    surf_point.y = extremum.y * t->step;
    surf_point.scale = 0.1333f * m->filter; // 1.2f / 9.0f;
    surf_point.hessian   = m->hessian[m_v];
    surf_point.laplacian = m->laplacian[m_v];
    surf_point.octave    = extremum.octave;
    surf_points_.push_back( surf_point );
  } else {
    // check the middle filter is mid way between top and bottom
//  assert( ( m->filter - b->filter ) > 0 and
//          t->filter - m->filter == m->filter - b->filter );

    // Get the offsets to the actual location of the extremum
    // Computes the partial derivatives in x, y, and scale of a pixel.
    // Computes the 3D Hessian matrix for a pixel.
    int x = extremum.x;
    int y = extremum.y;

    int t_cols = t->cols;
    int step = b->cols / t_cols;
    int col_step = step * b->cols;
    int index = col_step * y + step * x;
    int widthyx = t_cols * y + x;

    double dxs = ( t->hessian[ widthyx + 1 ] -
                   t->hessian[ widthyx - 1 ] -
                   b->hessian[ index + step ] +
                   b->hessian[ index - step ] ) / 4.0;

    double dys = ( t->hessian[ widthyx + t_cols ] -
                   t->hessian[ widthyx - t_cols ] -
                   b->hessian[ index + col_step ]  +
                   b->hessian[ index - col_step ] ) / 4.0;

    float da = t->hessian[ t_cols * y + x ];
    float db = b->hessian[ index ];

    step = m->cols / t_cols;
    col_step = step * m->cols;
    index = col_step * y  + step * x;
    double v = m->hessian[ index ];
    double dD[3];
    double H[3][3];
    dD[2] = ( da - db ) / 2.0;
    H[2][2] = da + db - 2 * v;
    da = m->hessian[ index + step ];
    db = m->hessian[ index - step ];
    dD[0] = ( da - db ) / 2.0;
    H[0][0] = da + db - 2 * v;
    da = m->hessian[ index + col_step ];
    db = m->hessian[ index - col_step ];
    dD[1] = ( da - db ) / 2.0;
    H[1][1] = da + db - 2 * v;

    double dxy = ( m->hessian[ index + col_step + step ] -
                   m->hessian[ index + col_step - step ] -
                   m->hessian[ index - col_step + step ] +
                   m->hessian[ index - col_step - step ] ) / 4.0;

    H[0][1] = dxy;
    H[0][2] = dxs;
    H[1][0] = dxy;
    H[1][2] = dys;
    H[2][0] = dxs;
    H[2][1] = dys;

    double H_inv[3][3];
    {
      cv::Mat tmp( 3, 3, CV_64F, H_inv );
      cv::invert( cv::Mat( 3, 3, CV_64F, H ), tmp, cv::DECOMP_SVD );
    }
    //  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );  // meaning H * X = dD
    double p[3] = {0};
    for ( int i = 0; i < 3; i++ ) {
      for ( int j = 0; j < 3; j++ ) {
        p[i] -= H_inv[i][j] * dD[j];
      }
    }

    // If point is sufficiently close to the actual extremum
//    statistics::record("laplacian_response", m->laplacian[m_v], m->hessian[m_v]); 
    if ( p[0] < 0.5f and p[0] > -0.5f and
         p[1] < 0.5f and p[1] > -0.5f and
         p[2] < 0.5f and p[2] > -0.5f 
         and m->laplacian[m_v] > 0
         ) {
//    statistics::record("interpolations", p[2]);
//    memset(p, 0, sizeof(double) * 3);
      SurfPoint surf_point;
      surf_point.x     = ( x + p[0] ) * t->step;
      surf_point.y     = ( y + p[1] ) * t->step;
      // 0.1333f = 1.2f / 9.0f = first_scale / first_filter_size
      surf_point.scale = 0.1333f * 
                        ( p[2] * ( m->filter - b->filter ) + m->filter );
      surf_point.hessian   = m->hessian[m_v];
      surf_point.laplacian = m->laplacian[m_v];
      surf_point.octave    = extremum.octave;
      surf_points_.push_back( surf_point );
//  } else {
//    std::cout << "interpolation failed " << std::endl;
    }
  }
}

//-------------------------------------------------------

void SmartSurf::compute( visual_odometry::KeyPoint * points,
                    int length,
                    bool upright ) {
  tellu( "KeyPoint" );
#ifdef VISUAL_ODOMETRY
#ifdef TIMER
  double time = cv::getTickCount();
#endif
  if ( !upright ) {
    for ( int i = 0; i < length; i++ ) {
      //       Assign Orientations and extract rotation invariant descriptors
      points[i].orientation = isInt() ?
                              getOrientation<int>( points[i].x,
                                                   points[i].y,
                                                   points[i].scale ) :
                              getOrientation<int64_t>( points[i].x,
                                                       points[i].y,
                                                       points[i].scale  );
    }
  }
  // Main Surf-64 loop assigns orientations and gets descriptors
#ifdef TIMER
  timer.e = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  for ( int i = 0; i < length; i++ ) {
    points[i].setDescriptor( 64, 0 );
    if ( isInt() ) {
      getDescriptor<int>( points[i].x,
                          points[i].y,
                          points[i].scale,
                          points[i].orientation,
                          points[i].getDescriptor() );
    } else {
      getDescriptor<int64_t>( points[i].x,
                              points[i].y,
                              points[i].scale,
                              points[i].orientation,
                              points[i].getDescriptor() );
    }
  }
#ifdef TIMER
  timer.f = cv::getTickCount() - time;
#endif
#endif // VISUAL_ODOMETRY
}

void SmartSurf::compute( const cv::Mat & source,
                    std::vector<cv::KeyPoint> & keys,
                    cv::Mat & descriptors ) {
  tellu( "cv::Mat" );
  init( source );
  compute( keys, descriptors );
}

void SmartSurf::compute( std::vector<cv::KeyPoint> & keys,
                    cv::Mat & descriptors ) {
  tellu( "std::vector<cv::KeyPoint>" );
  descriptors.create( keys.size(), 64, CV_32FC1 );
  float descriptor[64];
  for ( unsigned int i = 0; i < keys.size(); i++ ) {
    float angle = keys[i].angle * ( pi2 / 360.f );
    angle -= angle > pi ? pi2 : 0;
    if ( isInt() ) {
      getDescriptor<int>( keys[i].pt.x, keys[i].pt.y, keys[i].size,
                          angle, descriptor );
    } else {
      getDescriptor<int64_t>( keys[i].pt.x, keys[i].pt.y, keys[i].size,
                              angle, descriptor );
    }
    for ( int j = 0; j < 64; j++ ) {
      descriptors.at<float>( i, j ) = descriptor[j];
    }
  }
}

//-------------------------------------------------------

//! Assign the supplied Ipoint an orientation
template<class T> float SmartSurf::getOrientation( float key_point_x,
                                              float key_point_y,
                                              float key_point_scale ) {
  tellu( "" );
  T * data = (T *)integral_;
  int width = img_cols_ - 1;
  int height = img_cols_ * ( img_rows_ - 1 );
  const int scale = key_point_scale + 0.5f;
  //  std::cout << "scale " << scale << std::endl;
  const int scale_step = scale * img_cols_;
  const int scale_step_4 = 4 * scale_step;
  const int col = key_point_x - 0.5f;
  const int row = (int)( key_point_y - 0.5f ) * img_cols_;
  Angle angle[109];

  //  for ( int i=0; i<109; i++ ) assert(angle[i].x == 0 and angle[i].y == 0);

  int idx = -1;
  // calculate haar response for points within radius of 6*scale
  int c1 = col - 8 * scale;
  int c2 = col - 6 * scale;
  int c3 = col - 4 * scale;
  for ( int i = -5; i <= 5; i++ ) {

    int e = fastMin( 5, 8 + ( i > 0 ? -i : i ) );

    c1 = fastMin( c1 + scale, width );
    c2 = fastMin( c2 + scale, width );
    c3 = fastMin( c3 + scale, width );

    if ( c3 < 0 ) {
      idx += 2 * e + 1;
      // x and y are 0
      continue;
    }

    bool invalid_c1 = c1 < 0;
    int r1 = row - ( e + 3 ) * scale_step;

    for ( int j = -e; j <= e; j++ ) {
      idx++;
      r1       = fastMin( r1 + scale_step, height );
      bool invalid_r1 = r1 < 0;
      if ( r1 < -scale_step_4 ) {
        // x and y are 0
        continue;
      }

      int r3   = fastMin( r1 + scale_step_4, height );

      int A    = (                                  data[r3 + c3] ) -
                 (   invalid_r1 or invalid_c1 ? 0 : data[r1 + c1] );
      int B    = (   invalid_r1               ? 0 : data[r1 + c3] ) -
                 (                 invalid_c1 ? 0 : data[r3 + c1] );
      angle[idx].x = A - B + ( c2 < 0 ? 0 : 2 *
                               ( ( invalid_r1 ? 0 : data[r1 + c2] ) -
                                 (                  data[r3 + c2] ) ) );

      r3       = fastMin( r1 + scale_step + scale_step, height );
      angle[idx].y = A + B + ( r3 < 0  ? 0 : 2 *
                               ( ( invalid_c1 ? 0 : data[r3 + c1] ) -
                                 (                  data[r3 + c3] ) ) );
    }
  }

  for ( int i = 0; i < 109; i++ ) {
    float gauss = gauss_s1[i];
    angle[i].x *= gauss;
    angle[i].y *= gauss;
    angle[i].a = atan2( angle[i].y, angle[i].x );
  }

  std::sort( angle, angle + 109 );
  //  // calculate the dominant direction
  double max_sum = 0.f, max_sum_x = 0.f, max_sum_y = 0.f, sum_x = 0.f, sum_y = 0.f;
  int kdown = 0, kup = 0;
  float ang1 = -pi;
  // loop slides pi/3 window around feature point
  for (; ang1 < ( 5.11f - pi ); ang1 += 0.15f ) {    // 42 angles
    float ang2 = ang1 + pi / 3.0f;
    while ( kup < 109 and angle[kup].a <= ang2 ) {
      sum_x += angle[kup].x;
      sum_y += angle[kup].y;
      kup++;
    }
    while ( kdown < 109 and angle[kdown].a < ang1 ) {
      sum_x -= angle[kdown].x;
      sum_y -= angle[kdown].y;
      kdown++;
    }
    double tmp = sum_x * sum_x + sum_y * sum_y;
    if ( tmp > max_sum ) {
      max_sum = tmp;
      max_sum_x = sum_x;
      max_sum_y = sum_y;
    }
  }
  int n = 0;
  for (; angle[n].a <= 0 and n < 109; n++ ) angle[n].a += pi2;
  n += 109;
  for (; ang1 <= pi; ang1 += 0.15f ) {  // 42 angles
    float ang2 = ang1 + pi / 3.0f;
    //    float sum_x = 0.f, sum_y = 0.f;
    int k = kup % 109;
    while ( kup < n and angle[k].a <= ang2 ) {
      sum_x += angle[k].x;
      sum_y += angle[k].y;
      kup++;
      k = kup % 109;
    }
    while ( kdown < 109 and angle[kdown].a < ang1 ) {
      sum_x -= angle[kdown].x;
      sum_y -= angle[kdown].y;
      kdown++;
    }
    double tmp = sum_x * sum_x + sum_y * sum_y;
    if ( tmp > max_sum ) {
      max_sum = tmp;
      max_sum_x = sum_x;
      max_sum_y = sum_y;
    }
  }

  // assign orientation of the dominant response vector
  return max_sum ? atan2( max_sum_y, max_sum_x ) : 0;
}

//-------------------------------------------------------

//! Get the modified descriptor. See Agrawal ECCV 08
//! Modified descriptor contributed by Pablo Fernandez
template<class T> void SmartSurf::getDescriptor( float key_point_x,
                                            float key_point_y,
                                            float scale,
                                            float orientation,
                                            float * descriptor ) {
  tellu( "" );
  if ( !orientation ) {
    getDescriptorUpright<T>( key_point_x, key_point_y, scale, descriptor );
    return;
  }
  T * data = (T *)integral_;
  int width = img_cols_ - 1;
  int height = img_cols_ * ( img_rows_ - 1 );
  float co = cos( orientation );
  float si = sin( orientation );
  int sample[24][24][4];

  //Get coords of sample point on the rotated axis
#ifdef  OPEN_SURF_COMPATIBLE
  float x = (float)( (int)( key_point_x + 0.5f ) ) + 0.5f;
  float y = (float)( (int)( key_point_y + 0.5f ) ) + 0.5f;
  int * s = (int *)sample;
  for ( int i = -12; i < 12; i++ ) {
    for ( int j = -12; j < 12; j++ ) {
      s[0] = fastFloor( x + ( -j * scale * si + i * scale * co ) ) - 1;
      s[1] = fastFloor( y + ( j * scale * co + i * scale * si ) ) - 1;
      s += 4;
    }
  }
#else // OPEN_SURF_COMPATIBLE
  float x = (float)( (int)( key_point_x + 0.5f ) ) - 0.5f;
  float y = (float)( (int)( key_point_y + 0.5f ) ) - 0.5f;
  float scale_co = scale * co;
  float scale_si = scale * si;
  int * s = (int *)sample;
  {
    float scale_si_j[24];
    float scale_co_j[24];
    float sj = scale_si * -13;
    float cj = scale_co * -13;
    for ( int j = 0; j < 24; j++ ) {
      scale_si_j[j] = ( sj += scale_si );
      scale_co_j[j] = ( cj += scale_co );
    }
    float scale_co_i = scale_co * -13;
    float scale_si_i = scale_si * -13;
    for ( int i = -12; i < 12; i++ ) {
      scale_co_i += scale_co;
      scale_si_i += scale_si;
      for ( int j = 0; j < 24; j++ ) {
        s += 4;
        s[0] = fastFloor( x + scale_co_i    - scale_si_j[j] );
        s[1] = fastFloor( y + scale_co_j[j] + scale_si_i    );
      }
    }
  }
#endif // OPEN_SURF_COMPATIBLE

  int r_scale = scale + 0.5f;
  int s_scale = img_cols_ * r_scale;

  s = (int *)sample;
  for ( int i = 0; i < 24; i++ ) {
    for ( int j = 0; j < 24; j++ ) {
      if ( s[0] < -r_scale or s[1] < -r_scale ) {
        s[2] = s[3] = 0;
      } else {
        int row2 = s[1] * img_cols_;
        int row1 = fastMin( row2 - s_scale, height );
        int row3 = fastMin( row2 + s_scale, height );
        int c1   = fastMin( s[0] - r_scale,   width );
        int c2   = fastMin( s[0],             width );
        int c3   = fastMin( s[0] + r_scale,   width );
        bool invalid1 = row1 < 0;
        bool invalid3 = c1   < 0;

        int A = ( (                            data[row3 + c3] ) -
                  ( invalid1 or invalid3 ? 0 : data[row1 + c1] ) );
        int B = ( ( invalid1             ? 0 : data[row1 + c3] ) -
                  (             invalid3 ? 0 : data[row3 + c1] ) );
        s[2] = A - B + ( ( c2 < 0 ? 0 : 2 *
                           ( ( invalid1 ? 0 : data[row1 + c2] ) -
                             (                data[row3 + c2] ) ) ) );
        row2     = fastMin( row2,           height );
        s[3] = A + B + ( ( row2 < 0 ? 0 : 2 *
                           ( ( invalid3 ? 0 : data[row2 + c1] ) -
                             (                data[row2 + c3] ) ) ) );
      }
      s += 4;
    }
  }

  int count = 0;
#ifdef  OPEN_SURF_COMPATIBLE
  float sig = 2.5f * scale;
  float sig1 = 1.0f / ( pi * 2.0f * sig * sig );
#else
  float sig = -1.f / ( 12.5f * scale * scale );
#endif

  for ( int i = 0; i < 20; i += 5 ) {

    for ( int j = 0; j < 20; j += 5 ) {

      float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

      int xs, ys;
      {
#ifdef  OPEN_SURF_COMPATIBLE
        float ix = i - 7;
        float jx = j - 7;
        xs = fastFloor( x + ( -jx * scale * si + ix * scale * co ) ) - 1;
        ys = fastFloor( y + ( jx * scale * co + ix * scale * si ) ) - 1;
#else
        int ix = i - 7;
        int jx = j - 7;
        xs = fastFloor( scale_co * ix - scale_si * jx + x );
        ys = fastFloor( scale_co * jx + scale_si * ix + y );
#endif
      }

      for ( int k = i; k < i + 9; k++ ) {
        int * s = sample[k][j];

        //  float xx = ( 5 - k ) * scale_co - 5 * scale_si ;
        //  float yy = ( 5 - k ) * scale_si + 5 * scale_co ;

        for ( int l = 0; l < 9; l++ ) {
          int xx = xs - s[0];
          int yy = ys - s[1];

          //  xx += scale_si;
          //  yy += scale_co;
          //  int xx = ( fastFloor( x + ( ( -j   + 7  ) * scale * si +
          //                              (  i   - 7  ) * scale * co ) ) ) -
          //           ( fastFloor( x + ( ( -l-j + 12 ) * scale * si +
          //                              (  k   - 12 ) * scale * co ) ) );
          //  int yy = ( fastFloor( y + ( (  j   - 7  ) * scale * co +
          //                              (  i   - 7  ) * scale * si ) ) ) -
          //           ( fastFloor( y + ( (  l+j - 12 ) * scale * co +
          //                              (  k   - 12 ) * scale * si ) ) );


#ifdef  OPEN_SURF_COMPATIBLE
          float gauss = sig1 * exp( -( xx * xx + yy * yy ) /
                                    ( 2.0f * sig * sig ) );
#else
          // no sig1 because of normalization later on
          float gauss = exp( ( xx * xx + yy * yy ) * sig );
#endif
          float rrx = gauss * ( -s[2] * si + s[3] * co );
          float rry = gauss * (  s[2] * co + s[3] * si );

          //Get the gaussian weighted x and y response on rotated axis
          dx += rrx;
          dy += rry;
          mdx += rrx > 0 ? rrx : -rrx;
          mdy += rry > 0 ? rry : -rry;
          s += 4;
        }
      }

      //Add the values to the descriptor vector
#ifdef NEW_DESCRIPTOR
      descriptor[count] = dx;
      descriptor[count + 32] = mdx;
      count++;
      descriptor[count] = dy;
      descriptor[count + 32] = mdy;
      count++;
#else
      descriptor[count++] = dx;
      descriptor[count++] = dy;
      descriptor[count++] = mdx;
      descriptor[count++] = mdy;
#endif
    }
  }

  //Convert to Unit Vector
  float len = 0;
  int k = 0;
#ifdef  OPEN_SURF_COMPATIBLE
  for ( int i = 0; i < 16; i++ ) {
    float gauss = gauss_s2[i];
    len += ( descriptor[k]     * descriptor[k] +
             descriptor[k + 1] * descriptor[k + 1] +
             descriptor[k + 2] * descriptor[k + 2] +
             descriptor[k + 3] * descriptor[k + 3] ) * gauss * gauss;
    for ( int j = 0; j < 4; j++ ) {
      descriptor[k] *= gauss;
      k++;
    }
  }
  len = sqrt( len );
  for ( int i = 0; i < 64; i++ ) descriptor[i] /= len;
#else
#ifdef NEW_DESCRIPTOR
  float len1 = 0;
  float len2 = 0;
  for ( int i = 0; i < 16; i++ ) {
    for ( int j = 0; j < 2; j++ ) {
      descriptor[k     ] *= gauss_s2[i];
      descriptor[k + 32] *= gauss_s2[i];
      len1 += descriptor[k     ] * descriptor[k     ];
      len2 += descriptor[k + 32] * descriptor[k + 32];
      k++;
    }
  }
  len1 = 1.0f / sqrt( len1 );
  len2 = 1.0f / sqrt( len2 );
  for ( int i = 0; i < 32; i++ ) {
    descriptor[i     ] *= len1;
    descriptor[i + 32] *= len2;
  }
#else
  for ( int i = 0; i < 16; i++ ) {
    for ( int j = 0; j < 4; j++ ) {
      descriptor[k] *= gauss_s2[i];
      len += descriptor[k] * descriptor[k];
      k++;
    }
  }
  len = 1.0f / sqrt( len );
  for ( int i = 0; i < 64; i++ ) descriptor[i] *= len;
#endif
#endif

}

template<class T> void SmartSurf::getDescriptorUpright( float key_point_x,
                                                   float key_point_y,
                                                   float scale,
                                                   float * descriptor ) {
  tellu( "" );
  T * data = (T *)integral_;
  int width = img_cols_ - 1;
  int height = img_cols_ * ( img_rows_ - 1 );
  int sample[24][24][4];
  int r_scale = scale + 0.5f;
  int s_scale = img_cols_ * r_scale;
  int * s = (int *)sample;

#ifdef  OPEN_SURF_COMPATIBLE
  float x = (float)( (int)( key_point_x + 0.5f ) ) + 0.5f;
  float y = (float)( (int)( key_point_y + 0.5f ) ) + 0.5f;

  for ( int i = -12; i < 12; i++ ) {
    for ( int j = -12; j < 12; j++ ) {
      s[0] = fastFloor( x + i * scale ) - 1;
      s[1] = fastFloor( y + j * scale ) - 1;
#else
  float x = (float)( (int)( key_point_x + 0.5f ) ) - 0.5f;
  float y = (float)( (int)( key_point_y + 0.5f ) ) - 0.5f;
  float scale_i = scale * -13;

  for ( int i = -12; i < 12; i++ ) {
    scale_i += scale;
    float scale_j = -13 * scale;
    for ( int j = -12; j < 12; j++ ) {
      scale_j += scale;
      s[0] = fastFloor( x + scale_i );
      s[1] = fastFloor( y + scale_j );
#endif
      if ( s[0] < -r_scale or s[1] < -r_scale ) {
        s[2] = s[3] = 0;
      } else {
        int row2 = s[1] * img_cols_;
        int row1 = fastMin( row2 - s_scale, height );
        int row3 = fastMin( row2 + s_scale, height );
        int c1   = fastMin( s[0] - r_scale,   width );
        int c2   = fastMin( s[0],             width );
        int c3   = fastMin( s[0] + r_scale,   width );
        bool invalid1 = row1 < 0;
        bool invalid3 = c1   < 0;

        int A = ( (                            data[row3 + c3] ) -
                  ( invalid1 or invalid3 ? 0 : data[row1 + c1] ) );
        int B = ( ( invalid1             ? 0 : data[row1 + c3] ) -
                  (             invalid3 ? 0 : data[row3 + c1] ) );
        s[2] = A - B + ( ( c2 < 0 ? 0 : 2 *
                           ( ( invalid1 ? 0 : data[row1 + c2] ) -
                             (                data[row3 + c2] ) ) ) );
        row2     = fastMin( row2,           height );
        s[3] = A + B + ( ( row2 < 0 ? 0 : 2 *
                           ( ( invalid3 ? 0 : data[row2 + c1] ) -
                             (                data[row2 + c3] ) ) ) );
      }
      s += 4;
    }
  }

  int count = 0;
#ifdef  OPEN_SURF_COMPATIBLE
  float sig = 2.5f * scale;
  float sig1 = 1.0f / ( pi * 2.0f * sig * sig );
#else
  float sig = -1.f / ( 7.5f * scale * scale );
#endif

  for ( int i = 0; i < 20; i += 5 ) {

    for ( int j = 0; j < 20; j += 5 ) {

      float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

      int xs, ys;
      {
#ifdef  OPEN_SURF_COMPATIBLE
        float ix = i - 7;
        float jx = j - 7;
        xs = fastFloor( x + ix * scale ) - 1;
        ys = fastFloor( y + jx * scale ) - 1;
#else
        int ix = i - 7;
        int jx = j - 7;
        xs = fastFloor( scale * ix + x );
        ys = fastFloor( scale * jx + y );
#endif
      }

      for ( int k = i; k < i + 9; k++ ) {
        int * s = sample[k][j];

        //        float xx = ( 5 - k ) * scale;
        //        float yy = 5 * scale ;

        for ( int l = 0; l < 9; l++ ) {
          int xx = xs - s[0];
          int yy = ys - s[1];

          //          xx += scale;
          //          yy += scale;
          //          int xx = ( fastFloor( x + (  i   - 7  ) * scale ) ) -
          //                   ( fastFloor( x + (  k   - 12 ) * scale ) );
          //          int yy = ( fastFloor( y + (  j   - 7  ) * scale ) ) -
          //                   ( fastFloor( y + (  l+j - 12 ) * scale ) );

#ifdef  OPEN_SURF_COMPATIBLE
          float gauss = sig1 * exp( -( xx * xx + yy * yy ) /
                                    ( 2.0f * sig * sig ) );
#else
          // no sig1 because of normalization later on
          float gauss = exp( ( xx * xx + yy * yy ) * sig );
#endif
          float rrx = gauss * s[3];
          float rry = gauss * s[2];

          //Get the gaussian weighted x and y response on rotated axis
          dx += rrx;
          dy += rry;
          mdx += rrx > 0 ? rrx : -rrx;
          mdy += rry > 0 ? rry : -rry;
          s += 4;
        }
      }

      //Add the values to the descriptor vector
#ifdef NEW_DESCRIPTOR
      descriptor[count] = dx;
      descriptor[count + 32] = mdx;
      count++;
      descriptor[count] = dy;
      descriptor[count + 32] = mdy;
      count++;
#else
      descriptor[count++] = dx;
      descriptor[count++] = dy;
      descriptor[count++] = mdx;
      descriptor[count++] = mdy;
#endif
    }
  }

  //Convert to Unit Vector
  float len = 0;
  int k = 0;
#ifdef  OPEN_SURF_COMPATIBLE
  for ( int i = 0; i < 16; i++ ) {
    float gauss = gauss_s2[i];
    len += ( descriptor[k]     * descriptor[k] +
             descriptor[k + 1] * descriptor[k + 1] +
             descriptor[k + 2] * descriptor[k + 2] +
             descriptor[k + 3] * descriptor[k + 3] ) * gauss * gauss;
    for ( int j = 0; j < 4; j++ ) {
      descriptor[k] *= gauss_s2[i];
      k++;
    }
  }
  len = sqrt( len );
  for ( int i = 0; i < 64; i++ ) descriptor[i] /= len;
#else
#ifdef NEW_DESCRIPTOR
  float len1 = 0;
  float len2 = 0;
  for ( int i = 0; i < 16; i++ ) {
    for ( int j = 0; j < 2; j++ ) {
      descriptor[k     ] *= gauss_s2[i];
      descriptor[k + 32] *= gauss_s2[i];
      len1 += descriptor[k     ] * descriptor[k     ];
      len2 += descriptor[k + 32] * descriptor[k + 32];
      k++;
    }
  }
  len1 = 1.0f / sqrt( len1 );
  len2 = 1.0f / sqrt( len2 );
  for ( int i = 0; i < 32; i++ ) {
    descriptor[i     ] *= len1;
    descriptor[i + 32] *= len2;
  }
#else
  for ( int i = 0; i < 16; i++ ) {
    for ( int j = 0; j < 4; j++ ) {
      descriptor[k] *= gauss_s2[i];
      len += descriptor[k] * descriptor[k];
      k++;
    }
  }
  len = 1.0f / sqrt( len );
  for ( int i = 0; i < 64; i++ ) descriptor[i] *= len;
#endif
#endif

}

SmartSurf::Layer::Layer( int step,
                    int filter,
                    int cols,
                    int rows )  :
  step( step ),
  filter( filter ),
  cols( cols ),
  rows( rows ) {
  int size = this->cols * this->rows;
  hessian = new float[size];
  memset( hessian, 0, sizeof( float ) * size );
  laplacian = new float[size];
  memset( laplacian, 0, sizeof( float ) * size );
}

std::ostream & operator<<( std::ostream & out,
                           const SmartSurf & s ) {
  return out
         << "  SmartSurf:      " << std::endl
         << "    octaves:      " << s.octaves_ << std::endl
         << "    intervals:    " << s.intervals_ << std::endl
         << "    init step:    " << s.init_step_ << std::endl
         << "    use pyramide: " << ( s.use_pyramide_ ? "true" : "false" )
         << std::endl
         << "    interpolate:  " << ( s.interpolate_ ? "true" : "false" )
         << std::endl
         << "    threshold:    " << s.threshold_ << std::endl
         << "    img rows:     " << s.img_rows_ << std::endl
         << "    img columns:  " << s.img_cols_ << std::endl
         << "    layers:       " << s.pyramide_size_ << std::endl
         << "    key points:   " << s.size() << std::endl
#ifdef TIMER
         << "    timer (ms):   "
         << (int)( s.timer.a / 1e+6 ) << ", " << (int)( s.timer.b / 1e+6 ) << ", "
         << (int)( s.timer.c / 1e+6 ) << ", " << (int)( s.timer.d / 1e+6 ) << ", "
         << (int)( s.timer.e / 1e+6 ) << ", " << (int)( s.timer.f / 1e+6 ) << std::endl
#endif
  ;
}

}
