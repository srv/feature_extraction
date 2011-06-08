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

#include "opencv2/core/core.hpp"

#include "Surf.h"

#ifdef ODOMETRY
#include "Features.h"
#include "Debug.h"
#else
#define tellu( ... )
#endif

namespace odometry {

//***************************************************************
//*** Surf
//***************************************************************

const int Surf::OCTAVES;
const int Surf::INTERVALS;
const int Surf::INIT_STEP;
const float Surf::THRESHOLD_RESPONSE;
const int Surf::MAX_POINTS;
const int Surf::BOX_X;
const int Surf::BOX_Y;
const float Surf::DESCRIPTOR_SCALE_FACTOR;

const char * Surf::className = "Surf";
const int Surf::filter_map[OCTAVES][INTERVALS] = {{0, 1, 2, 3},
                                                   {1, 3, 4, 5},
                                                   {3, 5, 6, 7},
                                                   {5, 7, 8, 9},
                                                   {7, 9, 10, 11}};
#define pi 3.14159f
#define pi2 ( 3.14159f * 2 )
float Surf::gauss_s1[109] = {0};
float Surf::gauss_s2[16] = {0};

//-------------------------------------------------------

Surf::Surf( const cv::Mat & source,
            int octaves,
            int initStep,
            float threshold,
            int maxPoints,
            int boxX,
            int boxY ) :
  integral( 0 ),
  imgRows( 0 ),
  imgCols( 0 ),
  pyramide( 0 ),
  pyramideSize( 0 ) {
  init( source );
  init( octaves, initStep, threshold, maxPoints, boxX, boxY );
  //  std::cout << "imgRows " << imgRows
  //  << ", imgCols " << imgCols << std::endl;
}

Surf::Surf( int octaves,
            int initStep,
            float threshold,
            int maxPoints,
            int boxX,
            int boxY ) :
  integral( 0 ),
  imgRows( 0 ),
  imgCols( 0 ),
  pyramide( 0 ),
  pyramideSize( 0 ) {
  init( octaves, initStep, threshold, maxPoints, boxX, boxY );
}

void Surf::init( int octaves,
                 int initStep,
                 float threshold,
                 int maxPoints,
                 int boxX,
                 int boxY ) {
  tellu( "" );
  pyramide = 0;
  pyramideSize = 0;
  this->octaves = octaves > 0 and octaves < OCTAVES ? octaves : OCTAVES;
  this->initStep = initStep > 0 and initStep <= 6 ? initStep : INIT_STEP;
  this->thresholdResponse = threshold >= 0 ? threshold : THRESHOLD_RESPONSE;
  this->maxPoints = maxPoints > 0 ? maxPoints : MAX_POINTS;
  if ( !boxX ) boxX = maxPoints / 10;
  if ( boxY ) {
    this->boxX = boxX;
    this->boxY = boxY;
  } else {
    float fraq = imgRows and imgCols ? sqrt((float)imgCols / imgRows) : 1;
    this->boxX = sqrt(boxX) * fraq;
    this->boxY = sqrt(boxX) / fraq;
  }
//  std::cout << "x " << this->boxX << ", " << this->boxY << std::endl;
  //  float y1 = -10, y2 = 0, y3 = 10;
  //  for ( float x = -10; x < 20; x++ ) {
  //    printf( "% 2.3f, % 2.3f, % 2.3f :  % 2.3f, % 2.3f, % 2.3f\n",
  //            getAngle( x, y1 ), getAngle( x, y2 ), getAngle( x, y3 ),
  //           getAngle( y1, x ), getAngle( y2, x ), getAngle( y3, x ) );
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
    for ( int i = -5; i <= 5; ++i ) {
      int e = fastMin( 5, 8 + ( i > 0 ? -i : i ) );
      for ( int j = -e; j <= e; ++j ) {
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

void Surf::init( const cv::Mat & source ) {
  tellu( "" );
  if ( imgRows != source.rows or imgCols != source.cols ) {
    clear();
    imgRows = source.rows;
    imgCols = source.cols;
    if ( isInt() ) {
      integral = new int[imgRows * imgCols];
    } else {
      integral = new int64_t[imgRows * imgCols];
    }
  }
  if ( isInt() ) {
    integrate<int>( source );
  } else {
    integrate<int64_t>( source );
  }
}

void Surf::init( const cv::Mat & source,
                 const cv::Mat & ) {
  tellu( "" );
  init( source );
}

//-------------------------------------------------------

template<class T> void Surf::integrate( const cv::Mat & source ) {
  tellu( "" );
  T * data = (T *)integral;
  unsigned char * imgData;
  bool singleChannel = source.channels() == 1;
  if ( singleChannel ) {
    assert( source.isContinuous() );
    imgData = source.data;
  } else {
    imgData = new unsigned char[imgRows * imgCols];
    cv::Mat tmp( source.size(), CV_8UC1, imgData );
    cvtColor( source, tmp, CV_BGR2GRAY );
  }
  T sum = 0;
  int index = 0;
  int index2 = 0;
  for (; index < imgCols; index++ ) {
    sum += imgData[index];
    data[index] = sum;
  }
  for ( int i = 1; i < imgRows; ++i ) {
    sum = 0;
    for ( int j = 0; j < imgCols; ++j, index++, index2++ ) {
      sum += imgData[index];
      data[index] = data[index2] + sum;
    }
  }
  if ( !singleChannel ) delete[] imgData;
}

//-------------------------------------------------------

Surf::~Surf() {
  tellu( "" );
  clear();
}

void Surf::clear() {
  tellu( "" );
  if ( pyramide ) {
    for ( int i = 0; i < pyramideSize; ++i ) delete pyramide[i];
    delete[] pyramide;
    pyramide = 0;
    pyramideSize = 0;
  }
  if ( integral ) {
    if ( isInt() ) {
      delete[] (int *)integral;
    } else {
      delete[] (int64_t *)integral;
    }
    integral = 0;
    imgRows = 0;
    imgCols = 0;
  }
}

//-------------------------------------------------------

void Surf::operator()( const cv::Mat & source,
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

//! Find the image features and write into vector of features

void Surf::detect( const cv::Mat & source,
                   std::vector<cv::KeyPoint> & keys,
                   const cv::Mat & mask,
                   bool upright ) {
  tellu( "" );
  init( source, mask );
  detect( keys, upright );
}

void Surf::detect( std::vector<cv::KeyPoint> & keys,
                   bool upright ) {
  tellu( "" );
  detect();
  keys.clear();
  keys.resize( surfPoints.size() );
  for ( unsigned int i = 0; i < surfPoints.size(); i++ ) {
    keys[i].pt.x = surfPoints[i].x;
    keys[i].pt.y = surfPoints[i].y;
    keys[i].size = surfPoints[i].scale;
    keys[i].angle = upright ?   ( isInt() ?
                                  getOrientation<int>( surfPoints[i] ) :
                                  getOrientation<int64_t>( surfPoints[i] ) *
                                  ( 360.f / pi2 ) ) : 0;
    keys[i].angle += keys[i].angle < 0 ? 360.f : 0;
    keys[i].response = surfPoints[i].response;
  }
}

void Surf::detect() {
  tellu( "" );

  surfPoints.clear();
  extrema.clear();

#ifdef OpenSURFcpp
  double time = cv::getTickCount();
#endif
  buildPyramide();
#ifdef OpenSURFcpp
  timer.a = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  getExtrema();
  takeBest();
#ifdef OpenSURFcpp
  timer.b = cv::getTickCount() - time;
#endif
}

//-------------------------------------------------------

//! Build map of DoH responses
void Surf::buildPyramide() {
  tellu( "" );
  // Calculate responses for the first 4 octaves:
  // Oct1: 9,  15, 21, 27
  // Oct2: 15, 27, 39, 51
  // Oct3: 27, 51, 75, 99
  // Oct4: 51, 99, 147,195
  // Oct5: 99, 195,291,387

  int step = initStep;
  int width = imgCols / step;
  int height = imgRows / step;
  int m = 1;
  if (!pyramide)
  {
    pyramideSize = ( octaves + 1 ) * 2;
    pyramide = new Layer * [ pyramideSize ];
    pyramide[0] = new Layer( width, height, step, 2, m );
    pyramide[1] = new Layer( width, height, step, 4, m );

    for ( int i = 1; i <= octaves; i++ ) {
        pyramide[i * 2]     = new Layer( width, height, step, 6, m );
        pyramide[i * 2 + 1] = new Layer( width, height, step, 8, m );
        m *= 2;
    }
  }

  for ( int i = 0; i < pyramideSize; ++i ) {
    if ( isInt() ) {
      buildLayer<int>( pyramide[i] );
    } else {
      buildLayer<int64_t>( pyramide[i] );
    }
  }
}

//-------------------------------------------------------

//! Calculate DoH responses for supplied layer
template<class T> void Surf::buildLayer( Layer * layer ) {
  tellu( "" );
  T * data = (T *)integral;
  float * responses = layer->responses;         // response storage
  float * laplacian = layer->laplacian;         // laplacian storage
                                                //  int l = layer->filter;
                                                // filter size,  3*(2*n+1)
  int l2 = layer->filter / 2;                   // filer border,    3*n+1
  int l3 = layer->filter / 3;                   // filter lobe,     2*n+1
  int l6 = layer->filter / 6;                   //                    n
                                                // normalisation factor
  float inverse_area = 1.f / ( layer->filter * layer->filter );
  int imgColsStep = imgCols * layer->step;
  int l2Cols = l2 * imgCols;
  int l3Cols = l3 * imgCols;
  int l6Cols = l6 * imgCols;
  int height = imgCols * ( imgRows - 1 );
  int width = imgCols - 1;
  int index = 0;

#ifdef SMART_HESSIAN
  int ddsize = ( layer->width + l3 + 1 ) * ( layer->height + l3 + 1 );
  float DDxy( layer->width + l3 + 1 ) * ( layer->height + l3 + 1 );
  for ( int ar = 0, r = -imgCols;
        ar < layer->height + l3 + 1;
        ++ar, r += imgColsStep ) {

    for ( int ac = 0, c = -1;
          ac < layer->width + l3 + 1;
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
  for ( int ar = 0, r = 0;       //-imgCols;
        ar < layer->height;
        ++ar, r += imgColsStep ) {

    for ( int ac = 0, c = 0;       // -1;
          ac < layer->width;
          ++ac, c += layer->step ) {

      int r1a =        ( r - l3Cols                   );
      int r2a = fastMin( r + l3Cols - imgCols, height );
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
      r1a     =        ( r - l2Cols - imgCols         );
      r2a     = fastMin( r + l2Cols,           height );
      int r1b =        ( r - l6Cols - imgCols         );
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
      r1a     =        ( r - l3Cols - imgCols         );
      r2a     =        ( r          - imgCols         );
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
      int na = ( layer->width + l3 + 1 ) *   ar            + ac + l3  + 1;
      int nb = ( layer->width + l3 + 1 ) * ( ar + l3 + 1 ) + ac;
      int nc = ( layer->width + l3 + 1 ) *   ar            + ac;
      int nd = ( layer->width + l3 + 1 ) * ( ar + l3 + 1 ) + ac + l3 + 1;
      assert( DDxy[na] + DDxy[nb] - DDxy[nc] - DDxy[nd] == Dxy );
#endif
      
      // Normalise the filter responses with respect to their size
      Dxx *= inverse_area;
      Dyy *= inverse_area;
      Dxy *= inverse_area;

      // Get the determinant of hessian response & laplacian sign
      responses[index] = ( Dxx * Dyy - 0.81f * Dxy * Dxy );
      laplacian[index] = Dxx + Dyy;
      index++;
    }
  }
}

//-------------------------------------------------------

void Surf::getExtrema() {
  for ( int o = 0; o < octaves; ++o ) {
    for ( int i = 0; i <= 1; ++i ) {
      Layer * b = pyramide[filter_map[o][i]];
      Layer * m = pyramide[filter_map[o][i + 1]];
      Layer * t = pyramide[filter_map[o][i + 2]];
      //      std::cout << "response layers " << o << ", " << i << ": "
      //      << b->filter - 3 << ", " << m->filter - 3 << ", " << t->filter - 3 << "; "
      //      << t->filter + b->filter - 2 * m->filter << ", "
      //      << t->filter - b->filter << std::endl;
      int border = ( t->filter + 1 ) / ( 2 * t->step );
      int b_scale = b->width / t->width;
      int b_step = b->width * b_scale;
      int m_scale = m->width / t->width;
      int m_step = m->width * m_scale;
      int t_width = t->width;
      float * b_responses = b->responses;
      float * m_responses = m->responses;
      float * t_responses = t->responses;
      int box_height = t->height - border - border - 1;
      // loop over middle response layer at density of the most
      // sparse layer (always top), to find maxima across scale and space
      for ( int y = border + 1; y < t->height - border; ++y ) {
        int m_v = y * m_step + border * m_scale;
        int b_v = ( y - 1 ) * b_step + border * b_scale;
        int t_v = ( y - 1 ) * t_width;
        for ( int x = border + 1; x < t_width - border; ++x ) {
          m_v += m_scale;
          b_v += b_scale;
          float candidate = m_responses[ m_v ];
          if ( candidate < thresholdResponse ) continue;
          //-----------------------------------------------
          //! Non Maximal Suppression
          // check the candidate point in the middle layer is above threshold response
          int v = b_v;
          for ( int j = -1; j <= 1; ++j ) {
            if (
              b_responses[ v - b_scale ] >= candidate or
              b_responses[ v           ] >= candidate or
              b_responses[ v + b_scale ] >= candidate
              ) {
              goto no_extremum;
            }
            v += b_step;
          }
          v = m_v - m_step;
          for ( int j = -1; j <= 1; ++j ) {
            if (
              m_responses[ v - m_scale ] >= candidate or
              ( j and m_responses[ v ] >= candidate ) or
              m_responses[ v + m_scale ] >= candidate
              ) {
              goto no_extremum;
            }
            v += m_step;
          }
          v = t_v + x;
          for ( int j = -1; j <= 1; ++j ) {
            if (
              t_responses[ v - 1 ] >= candidate or
              t_responses[ v     ] >= candidate or
              t_responses[ v + 1 ] >= candidate
              ) {
              goto no_extremum;
            }
            v += t_width;
          }
          {
            int box = boxX *
                        std::max( 0.f,
                                  std::min( (float)boxY - 1,
                                            (float)boxY *
                                            ( y - border - 1 ) / box_height
                                            )
                                  ) +
                        std::max( 0.f,
                                  std::min( (float)boxX - 1,
                                            (float)boxX * ( x - border - 1 ) /
                                            ( t_width - border - border - 1 )
                                            )
                                  );
            extrema.push_back( Extremum( candidate,
                                         m->laplacian[ m_v ],
                                         x,
                                         y,
                                         box,
                                         filter_map[o][i],
                                         filter_map[o][i + 1],
                                         filter_map[o][i + 2] ) );
          }
          //-----------------------------------------------
no_extremum:;
          //-----------------------------------------------
        }
      }
      //      std::cout << "-- " << t->height - 2 * border - 1
      //      << ": " << b->height << ", "
      //      << m->height << ", "  << t->height << ", " << std::endl;
    }
  }
}


//! Interpolate scale-space extrema to subpixel accuracy to form an image feature.

void Surf::interpolate( Extremum & extremum ) {
  tellu( "" );
  int x = extremum.x;
  int y = extremum.y;
  Layer * t = pyramide[extremum.t];
  Layer * m = pyramide[extremum.m];
  Layer * b = pyramide[extremum.b];
  // check the middle filter is mid way between top and bottom
//  assert( ( m->filter - b->filter ) > 0 and
//          t->filter - m->filter == m->filter - b->filter );

  // Get the offsets to the actual location of the extremum
  // Computes the partial derivatives in x, y, and scale of a pixel.
  // Computes the 3D Hessian matrix for a pixel.

  int twidth = t->width;
  int scale = b->width / twidth;
  int scalewidth = scale * b->width;
  int index = scalewidth * y + scale * x;
  int widthyx = twidth * y + x;

  double dxs = ( t->responses[ widthyx + 1 ] -
                 t->responses[ widthyx - 1 ] -
                 b->responses[ index + scale ] +
                 b->responses[ index - scale ] ) / 4.0;

  double dys = ( t->responses[ widthyx + twidth ] -
                 t->responses[ widthyx - twidth ] -
                 b->responses[ index + scalewidth ]  +
                 b->responses[ index - scalewidth ] ) / 4.0;

  float da = t->responses[ twidth * y + x ];
  float db = b->responses[ index ];

  scale = m->width / twidth;
  scalewidth = scale * m->width;
  index = scalewidth * y  + scale * x;
  double v = m->responses[ index ];
  double dD[3];
  double H[3][3];
  dD[2] = ( da - db ) / 2.0;
  H[2][2] = da + db - 2 * v;
  da = m->responses[ index + scale ];
  db = m->responses[ index - scale ];
  dD[0] = ( da - db ) / 2.0;
  H[0][0] = da + db - 2 * v;
  da = m->responses[ index + scalewidth ];
  db = m->responses[ index - scalewidth ];
  dD[1] = ( da - db ) / 2.0;
  H[1][1] = da + db - 2 * v;

  double dxy = ( m->responses[ index + scalewidth + scale ] -
                 m->responses[ index + scalewidth - scale ] -
                 m->responses[ index - scalewidth + scale ] +
                 m->responses[ index - scalewidth - scale ] ) / 4.0;

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
  if ( p[0] < 0.5f and p[0] > -0.5f and
       p[1] < 0.5f and p[1] > -0.5f and
       p[2] < 0.5f and p[2] > -0.5f ) {
    SurfPoint surfPoint;
    surfPoint.x = ( x + p[0] ) * t->step;
    surfPoint.y = ( y + p[1] ) * t->step;
    surfPoint.scale = 0.1333f *
                      ( m->filter + p[2] * ( m->filter - b->filter ) );
    surfPoint.laplacian = extremum.laplacian;
    surfPoint.response  = extremum.response;
    surfPoints.push_back( surfPoint );
  }
}

#ifdef SURF_MARKER
static float ccc[10000] = {0};
static int ccc_counter = 0;
static float histogram[12] = {0};
#endif


//-------------------------------------------------------

float minScale[20] = {0};

void Surf::takeBest() {
  tellu( "" );
  int boxes = boxX * boxY;
#ifdef SURF_MARKER
  ccc_counter++;
  int mark_counter = 0;
  float ccount[boxes];
  memset( &ccount, 0, sizeof( float ) * boxes );
#endif
  surfPoints.clear();
  surfPoints.reserve( extrema.size() );
  int extrema_size = (int)extrema.size();
  bool taken[boxes];
  int count[boxes];
  memset( &taken, 0, sizeof( bool ) * boxes );
  memset( &count, 0, sizeof( int ) * boxes );
  for ( int i = 0; i < extrema_size; ++i ) {
    count[extrema[i].box]++;
#ifdef SURF_MARKER
    if ( true or extrema[i].m < 12 ) {
      ccount[extrema[i].box]++;
      mark_counter++;
    }
  }
  int c = 0;
  for ( int i = 0; i < boxY; i++ ) {
    for ( int j = 0; j < boxX; j++ ) {
      ccc[c] += ccount[c] / mark_counter;
      printf( "% 4i,", (int)( 10000 * ccc[c] / ccc_counter ) );
      c++;
    }
    printf( "\n" );
#endif
  }
  int toTake = maxPoints;
  int free = boxes;
  bool success = true;
  while ( success and free ) {
    success = false;
    int needed = toTake / free;
    for ( int i = 0; i < boxes; i++ ) {
      if ( !taken[i] and count[i] <= needed ) {
        free--;
        toTake -= count[i];
        taken[i] = true;
        success = true;
      }
    }
  }
  for ( int i = 0; i < extrema_size; i++ ) {
    if ( taken[extrema[i].box] ) interpolate( extrema[i] );
  }
  if ( !free ) return;
  int needed = toTake / free + ( toTake % free ? 1 : 0 );
  for ( int i = 0; i < boxes; i++ ) {
    if ( !taken[i] ) {
      Extremum myExtrema[count[i]];
      int c = 0;
      for ( int j = 0; j < extrema_size; j++ ) {
        if ( extrema[j].box == i ) {
          //          assert( c < count[i] );
          myExtrema[c] = extrema[j];
          c++;
        }
      }
      //      assert ( c == count[i] );
      sort( myExtrema, myExtrema + count[i] );
      for ( int j = 0; j < count[i] and j < needed; j++ ) {
        interpolate( myExtrema[j] );
      }
    }
  }
  //  for ( int i = 0; i < surfPoints.size(); i++ ) {
  //    for ( int j = 0; j < 20; j++ ) {
  //      if ( ( surfPoints[i].scale <  j + 2 ) and
  //           ( surfPoints[i].scale >  j + 1 ) ) minScale[j]++;
  //    }
  //  }
  //  for ( int j = 0; j < 20; j++ ) {
  //    std::cout << " ################## min scale = " << minScale[j] << std::endl;
  //  }
}


//! Describe all features in the supplied vector

void Surf::compute( const cv::Mat & source,
                    std::vector<cv::KeyPoint> & keys,
                    cv::Mat & descriptors ) {
  tellu( "" );
  init( source );
  compute( keys, descriptors );
}

void Surf::compute( std::vector<cv::KeyPoint> & keys,
                    cv::Mat & descriptors ) {
  tellu( "" );
  descriptors.create( keys.size(), 64, CV_32FC1 );
  float descriptor[64];
  for ( unsigned int i = 0; i < keys.size(); ++i ) {
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


#ifdef ODOMETRY
void Surf::compute( KeyPoint * points,
                    bool upright ) {
  tellu( "" );
  if ( !surfPoints.size() ) return;
  for ( unsigned int i = 0; i < surfPoints.size(); ++i ) {
    SurfPoint & surfPoint = surfPoints[i];
    points[i].set( surfPoint.x,
                   surfPoint.y,
                   surfPoint.scale,
                   0,
                   surfPoint.response,
                   surfPoint.laplacian,
                   64 );
  }
  // Get the size of the vector for fixed loop bounds
#ifdef OpenSURFcpp
  double time = cv::getTickCount();
#endif

  if ( !upright ) {
    for ( unsigned int i = 0; i < surfPoints.size(); ++i ) {
      //       Assign Orientations and extract rotation invariant descriptors
      points[i].orientation = isInt() ?
                              getOrientation<int>( surfPoints[i] ) :
                              getOrientation<int64_t>( surfPoints[i] );
    }
  }
  // Main Surf-64 loop assigns orientations and gets descriptors
#ifdef OpenSURFcpp
  timer.c = cv::getTickCount() - time;
  time = cv::getTickCount();
#endif
  for ( unsigned int i = 0; i < surfPoints.size(); ++i ) {
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
#ifdef OpenSURFcpp
  timer.d = cv::getTickCount() - time;
#endif
}

#endif

//-------------------------------------------------------

//! Assign the supplied Ipoint an orientation
template<class T> float Surf::getOrientation( SurfPoint & surfPoint ) {
  tellu( "" );
  T * data = (T *)integral;
  int width = imgCols - 1;
  int height = imgCols * ( imgRows - 1 );
  const int scale = DESCRIPTOR_SCALE_FACTOR * surfPoint.scale + 0.5f;
  //  std::cout << "scale " << scale << std::endl;
  const int scaleStep = scale * imgCols;
  const int scale4Step = 4 * scaleStep;
  const int row = (int)( surfPoint.y - 0.5f ) * imgCols;
  const int col = surfPoint.x - 0.5f;
  Angle angle[109];

  //  for ( int i=0; i<109; i++ ) assert(angle[i].x == 0 and angle[i].y == 0);

  int idx = -1;
  // calculate haar responses for points within radius of 6*scale
  int c1 = col - 8 * scale;
  int c2 = col - 6 * scale;
  int c3 = col - 4 * scale;
  for ( int i = -5; i <= 5; ++i ) {

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
    int r1 = row - ( e + 3 ) * scaleStep;

    for ( int j = -e; j <= e; ++j ) {
      idx++;
      r1       = fastMin( r1 + scaleStep, height );
      bool invalid_r1 = r1 < 0;
      if ( r1 < -scale4Step ) {
        // x and y are 0
        continue;
      }

      int r3   = fastMin( r1 + scale4Step, height );

      int A    = (                                  data[r3 + c3] ) -
                 (   invalid_r1 or invalid_c1 ? 0 : data[r1 + c1] );
      int B    = (   invalid_r1               ? 0 : data[r1 + c3] ) -
                 (                 invalid_c1 ? 0 : data[r3 + c1] );
      angle[idx].x = A - B + ( c2 < 0 ? 0 : 2 *
                               ( ( invalid_r1 ? 0 : data[r1 + c2] ) -
                                 (                  data[r3 + c2] ) ) );

      r3       = fastMin( r1 + scaleStep + scaleStep, height );
      angle[idx].y = A + B + ( r3 < 0  ? 0 : 2 *
                               ( ( invalid_c1 ? 0 : data[r3 + c1] ) -
                                 (                  data[r3 + c3] ) ) );
    }
  }

  for ( int i = 0; i < 109; i++ ) {
    float gauss = gauss_s1[i];
    angle[i].x *= gauss;
    angle[i].y *= gauss;
    angle[i].a = getAngle( angle[i].x, angle[i].y );
  }

  std::sort( angle, angle + 109 );

  //  // calculate the dominant direction
  double maxSum = 0.f, maxSumX = 0.f, maxSumY = 0.f, sumX = 0.f, sumY = 0.f;
  int kdown = 0, kup = 0;
  float ang1 = -pi;
  // loop slides pi/3 window around feature point
  for (; ang1 < ( 5.11f - pi ); ang1 += 0.15f ) {                     // 42 angles
    float ang2 = ang1 + pi / 3.0f;
    while ( kup < 109 and angle[kup].a <= ang2 ) {
      sumX += angle[kup].x;
      sumY += angle[kup].y;
      kup++;
    }
    while ( kdown < 109 and angle[kdown].a < ang1 ) {
      sumX -= angle[kdown].x;
      sumY -= angle[kdown].y;
      kdown++;
    }
    double tmp = sumX * sumX + sumY * sumY;
    if ( tmp > maxSum ) {
      maxSum = tmp;
      maxSumX = sumX;
      maxSumY = sumY;
    }
  }
  int n = 0;
  for (; angle[n].a <= 0 and n < 109; n++ ) angle[n].a += pi2;
  n += 109;
  for (; ang1 <= pi; ang1 += 0.15f ) {                         // 42 angles
    float ang2 = ang1 + pi / 3.0f;
    //    float sumX = 0.f, sumY = 0.f;
    int k = kup % 109;
    while ( kup < n and angle[k].a <= ang2 ) {
      sumX += angle[k].x;
      sumY += angle[k].y;
      kup++;
      k = kup % 109;
    }
    while ( kdown < 109 and angle[kdown].a < ang1 ) {
      sumX -= angle[kdown].x;
      sumY -= angle[kdown].y;
      kdown++;
    }
    double tmp = sumX * sumX + sumY * sumY;
    if ( tmp > maxSum ) {
      maxSum = tmp;
      maxSumX = sumX;
      maxSumY = sumY;
    }
  }

  // assign orientation of the dominant response vector
  return maxSum ? getAngle( maxSumX, maxSumY ) : 0;
}

//-------------------------------------------------------

//! Get the modified descriptor. See Agrawal ECCV 08
//! Modified descriptor contributed by Pablo Fernandez
template<class T> void Surf::getDescriptor( float keyPointX,
                                            float keyPointY,
                                            float keyPointScale,
                                            float orientation,
                                            float * descriptor ) {
  tellu( "" );
  if ( !orientation ) {
    getDescriptorUpright<T>( keyPointX, keyPointY, keyPointScale, descriptor );
    return;
  }
  T * data = (T *)integral;
  int width = imgCols - 1;
  int height = imgCols * ( imgRows - 1 );
  float co = cos( orientation );
  float si = sin( orientation );
  int sample[24][24][4];
#ifdef  OpenSURF_COMPATIBLE
  float scale = DESCRIPTOR_SCALE_FACTOR * keyPointScale;
  float x = (float)( (int)( keyPointX + 0.5f ) ) + 0.5f;
  float y = (float)( (int)( keyPointY + 0.5f ) ) + 0.5f;
#else
  float x = (float)( (int)( keyPointX + 0.5f ) ) - 0.5f;
  float y = (float)( (int)( keyPointY + 0.5f ) ) - 0.5f;
  float scale_co = keyPointScale * co;
  float scale_si = keyPointScale * si;
#endif

  //Get coords of sample point on the rotated axis
  int * s = (int *)sample;
#ifdef  OpenSURF_COMPATIBLE
  for ( int i = -12; i < 12; i++ ) {
    for ( int j = -12; j < 12; j++ ) {
      s[0] = fastFloor( x + ( -j * scale * si + i * scale * co ) ) - 1;
      s[1] = fastFloor( y + ( j * scale * co + i * scale * si ) ) - 1;
      s += 4;
    }
  }
#else
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
#endif

  int r_scale = keyPointScale + 0.5f;
  int s_scale = imgCols * r_scale;

  s = (int *)sample;
  for ( int i = 0; i < 24; i++ ) {
    for ( int j = 0; j < 24; j++ ) {
      if ( s[0] < -r_scale or s[1] < -r_scale ) {
        s[2] = s[3] = 0;
      } else {
        int row2 = s[1] * imgCols;
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
#ifdef  OpenSURF_COMPATIBLE
  float sig = 2.5f * keyPointScale;
  float sig1 = 1.0f / ( pi * 2.0f * sig * sig );
#else
  float sig = -1.f / ( 7.5f * keyPointScale * keyPointScale );
#endif

  for ( int i = 0; i < 20; i += 5 ) {

    for ( int j = 0; j < 20; j += 5 ) {

      float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

      int xs, ys;
      {
#ifdef  OpenSURF_COMPATIBLE
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

      for ( int k = i; k < i + 9; ++k ) {
        int * s = sample[k][j];

        //        float xx = ( 5 - k ) * scale_co - 5 * scale_si ;
        //        float yy = ( 5 - k ) * scale_si + 5 * scale_co ;

        for ( int l = 0; l < 9; ++l ) {
          int xx = xs - s[0];
          int yy = ys - s[1];

          //          xx += scale_si;
          //          yy += scale_co;

          //          int xx = ( fastFloor( x + ( ( -j   + 7  ) * scale * si +
          //                                      (  i   - 7  ) * scale * co ) ) ) -
          //                   ( fastFloor( x + ( ( -l-j + 12 ) * scale * si +
          //                                      (  k   - 12 ) * scale * co ) ) );
          //          int yy = ( fastFloor( y + ( (  j   - 7  ) * scale * co +
          //                                      (  i   - 7  ) * scale * si ) ) ) -
          //                   ( fastFloor( y + ( (  l+j - 12 ) * scale * co +
          //                                      (  k   - 12 ) * scale * si ) ) );


#ifdef  OpenSURF_COMPATIBLE
          float gauss = sig1 * exp( -( xx * xx + yy * yy ) /
                                    ( 2.0f * sig * sig ) );
#else
          // no sig1 because of normalization later on
          float gauss = exp( ( xx * xx + yy * yy ) * sig );
#endif
          float rrx = gauss * ( -s[2] * si + s[3] * co );
          float rry = gauss * (  s[2] * co + s[3] * si );

          //Get the gaussian weighted x and y responses on rotated axis
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
#ifdef  OpenSURF_COMPATIBLE
  for ( int i = 0; i < 16; ++i ) {
    float gauss = gauss_s2[i];
    len += ( descriptor[k]     * descriptor[k] +
             descriptor[k + 1] * descriptor[k + 1] +
             descriptor[k + 2] * descriptor[k + 2] +
             descriptor[k + 3] * descriptor[k + 3] ) * gauss * gauss;
    for ( int j = 0; j < 4; ++j ) {
      descriptor[k] *= gauss_s2[i];
      k++;
    }
  }
  len = sqrt( len );
  for ( int i = 0; i < 64; ++i ) descriptor[i] /= len;
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
  for ( int i = 0; i < 32; ++i ) {
    descriptor[i     ] *= len1;
    descriptor[i + 32] *= len2;
  }
#else
  for ( int i = 0; i < 16; ++i ) {
    for ( int j = 0; j < 4; ++j ) {
      descriptor[k] *= gauss_s2[i];
      len += descriptor[k] * descriptor[k];
      k++;
    }
  }
  len = 1.0f / sqrt( len );
  for ( int i = 0; i < 64; ++i ) descriptor[i] *= len;
#endif
#endif

}

template<class T> void Surf::getDescriptorUpright( float keyPointX,
                                                   float keyPointY,
                                                   float keyPointScale,
                                                   float * descriptor ) {
  tellu( "" );
  T * data = (T *)integral;
  int width = imgCols - 1;
  int height = imgCols * ( imgRows - 1 );
  int sample[24][24][4];
  float scale = DESCRIPTOR_SCALE_FACTOR * keyPointScale;
#ifdef  OpenSURF_COMPATIBLE
  float x = (float)( (int)( keyPointX + 0.5f ) ) + 0.5f;
  float y = (float)( (int)( keyPointY + 0.5f ) ) + 0.5f;
#else
  float x = (float)( (int)( keyPointX + 0.5f ) ) - 0.5f;
  float y = (float)( (int)( keyPointY + 0.5f ) ) - 0.5f;
#endif

#ifndef OpenSURF_COMPATIBLE
  float scale_i = scale * -13;
#endif
  int r_scale = keyPointScale + 0.5f;
  int s_scale = imgCols * r_scale;
  int * s = (int *)sample;

  for ( int i = -12; i < 12; i++ ) {
#ifndef OpenSURF_COMPATIBLE
    scale_i += scale;
    float scale_j = -13 * scale;
#endif
    for ( int j = -12; j < 12; j++ ) {
#ifdef  OpenSURF_COMPATIBLE
      s[0] = fastFloor( x + i * scale ) - 1;
      s[1] = fastFloor( y + j * scale ) - 1;
#else
      scale_j += scale;
      s[0] = fastFloor( x + scale_i );
      s[1] = fastFloor( y + scale_j );
#endif
      if ( s[0] < -r_scale or s[1] < -r_scale ) {
        s[2] = s[3] = 0;
      } else {
        int row2 = s[1] * imgCols;
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
#ifdef  OpenSURF_COMPATIBLE
  float sig = 2.5f * keyPointScale;
  float sig1 = 1.0f / ( pi * 2.0f * sig * sig );
#else
  float sig = -1.f / ( 7.5f * keyPointScale * keyPointScale );
#endif

  for ( int i = 0; i < 20; i += 5 ) {

    for ( int j = 0; j < 20; j += 5 ) {

      float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

      int xs, ys;
      {
#ifdef  OpenSURF_COMPATIBLE
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

      for ( int k = i; k < i + 9; ++k ) {
        int * s = sample[k][j];

        //        float xx = ( 5 - k ) * scale;
        //        float yy = 5 * scale ;

        for ( int l = 0; l < 9; ++l ) {
          int xx = xs - s[0];
          int yy = ys - s[1];

          //          xx += scale;
          //          yy += scale;
          //          int xx = ( fastFloor( x + (  i   - 7  ) * scale ) ) -
          //                   ( fastFloor( x + (  k   - 12 ) * scale ) );
          //          int yy = ( fastFloor( y + (  j   - 7  ) * scale ) ) -
          //                   ( fastFloor( y + (  l+j - 12 ) * scale ) );

#ifdef  OpenSURF_COMPATIBLE
          float gauss = sig1 * exp( -( xx * xx + yy * yy ) /
                                    ( 2.0f * sig * sig ) );
#else
          // no sig1 because of normalization later on
          float gauss = exp( ( xx * xx + yy * yy ) * sig );
#endif
          float rrx = gauss * s[3];
          float rry = gauss * s[2];

          //Get the gaussian weighted x and y responses on rotated axis
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
#ifdef  OpenSURF_COMPATIBLE
  for ( int i = 0; i < 16; ++i ) {
    float gauss = gauss_s2[i];
    len += ( descriptor[k]     * descriptor[k] +
             descriptor[k + 1] * descriptor[k + 1] +
             descriptor[k + 2] * descriptor[k + 2] +
             descriptor[k + 3] * descriptor[k + 3] ) * gauss * gauss;
    for ( int j = 0; j < 4; ++j ) {
      descriptor[k] *= gauss_s2[i];
      k++;
    }
  }
  len = sqrt( len );
  for ( int i = 0; i < 64; ++i ) descriptor[i] /= len;
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
  for ( int i = 0; i < 32; ++i ) {
    descriptor[i     ] *= len1;
    descriptor[i + 32] *= len2;
  }
#else
  for ( int i = 0; i < 16; ++i ) {
    for ( int j = 0; j < 4; ++j ) {
      descriptor[k] *= gauss_s2[i];
      len += descriptor[k] * descriptor[k];
      k++;
    }
  }
  len = 1.0f / sqrt( len );
  for ( int i = 0; i < 64; ++i ) descriptor[i] *= len;
#endif
#endif

}

}
