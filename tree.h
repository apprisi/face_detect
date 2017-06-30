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




#ifndef _BINARY_TREE_H_
#define _BINARY_TREE_H_

#include "sample.h"
#include <omp.h>


typedef struct {
    uint8_t x0, y0;
    uint8_t x1, y1;

    uint8_t w, h;
} FeatTemp;


void print_feature_template(FeatTemp *ft, FILE *fout);


typedef struct Node_t{
    float thresh;
    float score;

    FeatTemp ft;

    struct Node_t *lchild;
    struct Node_t *rchild;

    //for debug
    double nw, pw;
    int posSize, negSize;
} Node;


typedef Node Tree;



float predict(Tree *root, int depth, uint32_t *iImg, int stride);

float train(Tree* root, int depth, SampleSet *posSet, SampleSet *negSet, float recall);

void print_tree(Tree* root, int depth, FILE *fout);

void save(Tree* root, int depth, FILE *fout);
void load(Tree* root, int depth, FILE *fin);

#endif
