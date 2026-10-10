#ifndef INCLUDE_DATA_STRUCTURES_H
#define INCLUDE_DATA_STRUCTURES_H

#include <opencv2/core.hpp>
#include <vector>

struct BoundingBox {  // bounding box around a classified object (contains both
                      // 2D and 3D data)

  int boxID;    // unique identifier for this bounding box
  int trackID;  // unique identifier for the track to which this bounding box
                // belongs

  cv::Rect roi;       // 2D region-of-interest in image coordinates
  int classID;        // ID based on class file provided to YOLO framework
  double confidence;  // classification trust
};

struct DataFrame {  // represents the available sensor information at the same
                    // time instance

  cv::Mat cameraImg_l;  // left camera image
  cv::Mat cameraImg_c;  // center camera image
  cv::Mat cameraImg_r;  // right camera image

  std::vector<BoundingBox>
      boundingBoxes;  // ROI around detected objects in 2D image coordinates
};

#endif /* INCLUDE_DATA_STRUCTURES_H */
