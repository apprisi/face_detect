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

#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include "tool.h"

typedef struct {
    uint8_t *img;
    int stride;

    uint32_t *iImgBuf;
    uint32_t *iImg;
    int istride;

    float score;

    char patchName[100];
} Sample;


void release_data(Sample *sample);
void release(Sample **sample);


typedef struct{
    Sample **samples;
    int ssize;
    int winw;
    int winh;

    int capacity;
} SampleSet;


int read_samples(const char *fileList, int ssize, int WINW, int WINH, int mirrorFlag, SampleSet **posSet);

int save(const char *filePath, SampleSet *set);
int load(const char *filePath, SampleSet **set);

void reserve(SampleSet *set, int size);

void create_sample(Sample *sample, uint8_t *img, int width, int height, int stride, const char *patchName);

void add_sample_capacity_unchange(SampleSet *set, Sample *sample);


void random_order(SampleSet* set);

void write_images(SampleSet *set, const char *outDir, int step);

void release_data(SampleSet *set);
void release(SampleSet **set);

void print_info(SampleSet *set, const char *tag);

#endif
