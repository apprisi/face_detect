/*M///////////////////////////////////////////////////////////////////////////////////////////////////
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//    * Using for Personal applications or study are freedom.
//    * Using for Bussiness must ask for agreements.
//
//  Copyright (C) 2017, Hangzhou Qiantu Technology Copyright,(杭州千图科技有限公司) all rights reserved.
//  Author: Zhu xiaoyan
//  email: business@qiantuai.com
//  website: www.qiantuai.com
//M*/


#ifndef _TOOL_H_
#define _TOOL_H_

#include "typedef.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int read_list(const char *filePath, std::vector<std::string> &fileList);
void split_file_path(const char* filePath, char *rootDir, char *fileName, char *ext);

void QT_integral_image(uint8_t *img, int width, int height, int stride, uint32_t *intImg, int istride);
void QT_resize_gray_image(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts);
void update_weights(double *weights, int size);

void quick_sort_float(float *arr, int size);
void transform_image(cv::Mat &img, int WINW);
void transform_image(uint8_t *img, int width, int height, int stride, uint8_t *dImg);

void horizontal_mirror(uint8_t *img, int width, int height, int stride);
void vertical_mirror(uint8_t *img, int width, int height, int stride);

void sleep(uint64_t milisecond);
#endif
