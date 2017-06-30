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


#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "tree.h"
#include <omp.h>

#define NEG_IMAGES_FILE "neg_images.bin"

typedef struct {
    float recall;

    int treeSize;
    int depth;
    int flag;
} TrainParams;


typedef struct {
    Tree **trees;

    int treeSize;

    int capacity;
    int depth;

    float *threshes;
} StrongClassifier;


typedef struct{
    FILE *fin;
    int isize;
    int id;

    StrongClassifier *scs;
    int scSize;

    int dx, dy;
    int tflag;

    int npRate;

    int maxCount;
} NegGenerator;


int generate_negative_images(const char *listFile, const char *outfile);

void train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, NegGenerator *generator, TrainParams *params);

int predict(StrongClassifier *sc, uint32_t *intImg, int istride, float &score);
int predict(StrongClassifier *sc, int scSize, uint32_t *intImg, int istride, float &score);

int save(StrongClassifier *sc, FILE *fout);
int load(StrongClassifier *sc, FILE *fin);

void release(StrongClassifier **sc);
void release_data(StrongClassifier *sc);

#endif
