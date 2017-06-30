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



#ifndef _ADA_BOOST_H_
#define _ADA_BOOST_H_

#include "classifier.h"


typedef struct{
    StrongClassifier *sc;
    int WINW, WINH;
    int ssize;

    float startScale;
    float endScale;
    float offsetFactor;

    int layer;
}QTObjectDetector;


typedef struct {
    int x, y;
    int width;
    int height;
} QTRect;

int train(QTObjectDetector *cc, const char *posFilePath, const char *negFilePath);
int predict(QTObjectDetector *cc, uint32_t *iImg, int iStride, float &score);

void init_detect_factor(QTObjectDetector *cc, float startScale, float endScale, float offset, int layer);
int detect(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride, QTRect **resRect, float **rscores);

int save(QTObjectDetector *cc, const char *filePath);
int load(QTObjectDetector *cascade, const char *filePath);

void release_data(QTObjectDetector *cc);
void release(QTObjectDetector **cc);

void release_data(NegGenerator *ng);
void release(NegGenerator **ng);

#endif
