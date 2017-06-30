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


#include "sample.h"


//用给出的图像生成样本
void create_sample(Sample *sample, uint8_t *img, int width, int height, int stride, const char *patchName){
    if(sample->img == NULL)
        sample->img = new uint8_t[width * height];

    if(sample->iImgBuf == NULL)
        sample->iImgBuf = new uint32_t[(width + 1) * (height + 1)];

    sample->iImg = sample->iImgBuf + (width + 1) + 1;
    sample->istride = width + 1;
    sample->stride = width;
    sample->score = 0;

    strcpy(sample->patchName, patchName);

    for(int y = 0; y < height; y++)
        memcpy(sample->img + y * width, img + y * stride, sizeof(uint8_t) * width);

    memset(sample->iImgBuf, 0, sizeof(uint32_t) * (width + 1) * (height + 1));

    QT_integral_image(sample->img, width, height, width, sample->iImg, sample->istride);
}


void release_data(Sample *sample){
    if(sample->img != NULL)
        delete [] sample->img;
    sample->img = NULL;

    if(sample->iImgBuf != NULL)
        delete [] sample->iImgBuf;

    sample->iImgBuf = NULL;
    sample->iImg = NULL;
}


void release(Sample **sample){
    if(*sample != NULL){
        release_data(*sample);
        delete *sample;
    }

    *sample = NULL;
}


int read_samples(const char *fileList, int ssize, int WINW, int WINH, int mirrorFlag, SampleSet **posSet){
    std::vector<std::string> imgList;
    char rootDir[256], fileName[256], ext[30];
    int ret;

    if(load("pos_samples.bin", posSet) == 0){
        printf("POSITIVE SAMPLE SIZE: %d\n", (*posSet)->ssize);
        return (*posSet)->ssize;
    }

    if(read_list(fileList, imgList) != 0)
        return 0;

    if(ssize <= 100)
        ssize = imgList.size();
    else
        ssize = QT_MIN(ssize, imgList.size());

    if(ssize <= 0) return 0;

    *posSet = new SampleSet;

    (*posSet)->winw = WINW;
    (*posSet)->winh = WINH;

    if(mirrorFlag == 1)
        (*posSet)->ssize = ssize * 2;
    else
        (*posSet)->ssize = ssize;

    (*posSet)->samples = new Sample*[(*posSet)->ssize];

    memset((*posSet)->samples, 0, sizeof(Sample*) * (*posSet)->ssize);

    for(int i = 0; i < ssize; i++){
        const char *imgPath = imgList[i].c_str();

        cv::Mat img = cv::imread(imgPath, 0);
        assert(!img.empty());

        split_file_path(imgPath, rootDir, fileName, ext);

        Sample *sample = new Sample;

        sample->img = new uint8_t[WINW * WINH];
        sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
        sample->iImg = sample->iImgBuf + (WINW + 1) + 1;
        sample->istride = WINW + 1;
        sample->stride = WINW;

        sprintf(sample->patchName, "%s", fileName);

        QT_resize_gray_image(img.data, img.cols, img.rows, img.step,
                sample->img, WINW, WINH, WINW);

        memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

        QT_integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);

        if(mirrorFlag == 0){
            (*posSet)->samples[i] = sample;
        }
        else {
            (*posSet)->samples[i * 2] = sample;

            //horizontal
            sample = new Sample;

            sample->img = new uint8_t[WINW * WINH];
            sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
            sample->iImg = sample->iImgBuf + (WINW + 1) + 1;
            sample->istride = WINW + 1;
            sample->stride = WINW;

            sprintf(sample->patchName, "%s_h", fileName);

            memcpy(sample->img, (*posSet)->samples[i * 2]->img, sizeof(uint8_t) * WINW * WINH);

            horizontal_mirror(sample->img, WINW, WINH, WINW);

            memset(sample->iImgBuf, 0, sizeof(uint32_t) * (WINW + 1) * (WINH + 1));

            QT_integral_image(sample->img, WINW, WINH, WINW, sample->iImg, WINW + 1);

            (*posSet)->samples[i * 2 + 1] = sample;
        }

        printf("%d\r", i + 1); fflush(stdout);
    }

    (*posSet)->capacity = (*posSet)->ssize;

    save("pos_samples.bin", *posSet);

    return ssize;
}


int save(const char *filePath, SampleSet *set){
    FILE *fout = fopen(filePath, "wb");

    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;

    int WINW = set->winw;
    int WINH = set->winh;

    ret = fwrite(&set->ssize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->winw, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->winh, sizeof(int), 1, fout); assert(ret == 1);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        ret = fwrite(sample->img, sizeof(uint8_t), WINW * WINH, fout); assert(ret == WINW * WINH);
        ret = fwrite(sample->patchName, sizeof(char), 100, fout); assert(ret == 100);
    }

    fclose(fout);

    return 0;
}


int load(const char *filePath, SampleSet **resSet){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;
    int WINW, WINH, ssize;

    SampleSet *set = new SampleSet;
    *resSet = set;

    memset(set, 0, sizeof(SampleSet));


    ret = fread(&ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->winw, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->winh, sizeof(int), 1, fin); assert(ret == 1);

    printf("%d %d %d\n", ssize, set->winw, set->winh);

    WINW = set->winw;
    WINH = set->winh;

    reserve(set, ssize);

    for(int i = 0; i < ssize; i++){
        Sample *sample = new Sample;
        set->samples[i] = sample;

        memset(sample, 0, sizeof(Sample));

        sample->img = new uint8_t[WINW * WINH];
        sample->iImgBuf = new uint32_t[(WINW + 1) * (WINH + 1)];
        sample->iImg = sample->iImgBuf + WINW + 1 + 1;

        sample->stride = WINW;
        sample->istride = WINW + 1;

        sample->score = 0;

        ret = fread(sample->img, sizeof(uint8_t), WINW * WINH, fin); assert(ret == WINW * WINH);
        ret = fread(sample->patchName, sizeof(char), 100, fin); assert(ret == 100);

        QT_integral_image(sample->img, WINW, WINH, WINW, sample->iImg, sample->istride);

    }

    set->ssize = ssize;

    fclose(fin);

    return 0;
}



void add_sample(SampleSet *set, Sample *sample){
    if(set->ssize == set->capacity){
        if(set->ssize > 0){
            Sample **samples = new Sample*[set->ssize * 2];

            memcpy(samples, set->samples, sizeof(Sample*) * set->ssize);
            memset(samples + set->ssize, 0, sizeof(Sample*) * set->ssize);
            memset(set->samples, 0, sizeof(Sample*) * set->ssize);

            delete [] set->samples;

            set->samples = samples;
            set->capacity = set->ssize * 2;
        }
        else {
            set->samples = new Sample*[100];
            memset(set->samples, 0, sizeof(Sample*) * 100);
            set->capacity = 100;
        }
    }

    set->samples[set->ssize] = sample;
    set->ssize ++;
}


//扩大SampleSet元素空间
void reserve(SampleSet *set, int size){
    if(set->capacity > size)
        return ;

    Sample **samples = new Sample*[size];

    memset(samples, 0, sizeof(Sample*) * size);

    if(set->ssize > 0){
        memcpy(samples, set->samples, sizeof(Sample*) * set->ssize);
        memset(set->samples, 0, sizeof(Sample*) * set->ssize);
    }

    if(set->samples != NULL)
        delete [] set->samples;

    set->samples = samples;
    set->capacity = size;
}


void add_sample_capacity_unchange(SampleSet *set, Sample *sample){
    if(set->capacity <= set->ssize){
        cv::RNG rng(cv::getTickCount());
        int len = 0.1 * set->ssize + 1;

        for(int i = 0; i < len; i++){
            int id = rng.uniform(0, set->ssize);

            QT_SWAP(set->samples[id], set->samples[set->ssize - 1], Sample*);
            release(&set->samples[set->ssize - 1]);

            set->ssize --;
        }
    }

    set->samples[set->ssize++] = sample;
}


void random_order(SampleSet* set){
    cv::RNG rng(cv::getTickCount());

    for(int i = 0; i < set->ssize; i++){
        int id1 = rng.uniform(0, set->ssize);
        int id2 = rng.uniform(0, set->ssize);

        if(id1 == id2) continue;

        QT_SWAP(set->samples[id1], set->samples[id2], Sample*);
    }
}


void release_data(SampleSet *set){
    if(set != NULL){
        if(set->samples != NULL){
            for(int i = 0; i < set->capacity; i++){
                if(set->samples[i] != NULL)
                    release(&set->samples[i]);

                set->samples[i] = NULL;
            }

            delete [] set->samples;
        }

        set->samples = NULL;

        set->ssize = 0;
        set->capacity = 0;
    }
}


void release(SampleSet **set){
    release_data(*set);

    delete *set;
    *set = NULL;
}


void print_info(SampleSet *set, const char *tag){
    printf("SampleSet: %s\n", tag);

    printf("ssize: %d, winw: %d, winh: %d\n", set->ssize, set->winw, set->winh);
    printf("capacity: %d\n", set->capacity);
}


void write_images(SampleSet *set, const char *outDir, int step){
    int ssize = set->ssize;

    int WINW = set->winw;
    int WINH = set->winh;

    char outPath[256], command[256];

    sprintf(command, "mkdir -p %s", outDir);

    int ret = system(command);

    for(int i = 0; i < ssize; i+= step){
        Sample *sample = set->samples[i];
        cv::Mat img(WINH, WINW, CV_8UC1, sample->img, sample->stride);

        sprintf(outPath, "%s/%s.jpg", outDir, sample->patchName);
        if(!cv::imwrite(outPath, img))
            printf("Can't write image %s\n", outPath);

        printf("%d\r", i); fflush(stdout);
    }
}
