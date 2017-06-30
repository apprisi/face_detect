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
/
//  Copyright (C) 2017, Hangzhou Qiantu Technology Copyright,(杭州千图科技有限公司) all rights reserved.
//  Author: Zhu xiaoyan
//  email: business@qiantuai.com
//  website: www.qiantuai.com
//M*/



#include "adaboost.h"


void refine_samples(QTObjectDetector *cc, SampleSet *set, int flag);

void init_train_params(TrainParams *params, NegGenerator *generator, int WINW, int WINH, int s){
    generator->dx = 0.1f * WINW;
    generator->dy = 0.1f * WINH;
    generator->tflag = 1;
    generator->maxCount = 20;
    generator->npRate = 1;

    params->recall = 1.0f;
    params->flag = 1;
    params->depth = 4;

    switch(s){
        case 0:
            generator->dx = WINW;
            generator->dy = WINH;
            generator->npRate = 20;
            generator->maxCount = 10;
            generator->tflag = 0;

            params->treeSize = 128;
            params->recall = 0.9999f;

            break;

        case 1:
            generator->dx = WINW * 0.3;
            generator->dy = WINH * 0.3;
            generator->npRate = 1;
            generator->maxCount = 10;

            params->treeSize = 192;

            break;

        case 2:
            generator->npRate = 1;
            generator->maxCount = 10;

            params->treeSize = 192;

            break;

        case 3:
            generator->npRate = 1;
            generator->maxCount = 20;

            params->treeSize = 256;
            params->flag = 0;

            break;

        case 4:
            params->treeSize = 256;
            params->flag = 0;

            break;

        case 5:
            params->treeSize = 320;
            params->flag = 0;

            break;

        default:
            break;
    }
}


int train(QTObjectDetector *cc, const char *posFilePath, const char *negFilePath){
    const int WINW = 64;
    const int WINH = 64;

    const int STAGE = 6;

    SampleSet *posSet = NULL, *negSet = NULL;

    TrainParams params;
    NegGenerator generator;

    int ret;

    for(int i = STAGE - 1; i >= 0; i--){
        char filePath[256];

        sprintf(filePath, "cascade_%d.dat", i);

        if(load(cc, filePath) == 0){
            printf("LOAD MODEL %s SUCCESS\n", filePath);
            break;
        }
    }

    if(cc->ssize == 0){
        cc->WINW = WINW;
        cc->WINH = WINH;

        cc->sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(cc->sc, 0, sizeof(StrongClassifier) * STAGE);
    }
    else {
        StrongClassifier *sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(sc, 0, sizeof(StrongClassifier) * STAGE);
        memcpy(sc, cc->sc, sizeof(StrongClassifier) * cc->ssize);

        delete [] cc->sc;
        cc->sc = sc;
    }

    read_samples(posFilePath, 0, WINW, WINH, 0, &posSet);
    print_info(posSet, "pos set");

    negSet = new SampleSet;

    memset(negSet, 0, sizeof(SampleSet));

    negSet->winw = WINW;
    negSet->winh = WINH;

    generate_negative_images(negFilePath, NEG_IMAGES_FILE);

    generator.fin = fopen(NEG_IMAGES_FILE, "rb");
    assert(generator.fin != NULL);

    ret = fread(&generator.isize, sizeof(int), 1, generator.fin);assert(ret == 1);
    generator.id = 0;

    ret = system("mkdir -p model log/neg log/pos");

    for(int s = cc->ssize; s < STAGE; s++){
        printf("---------------- CASCADE %d ----------------\n", s);
        init_train_params(&params, &generator, WINW, WINH, s);

        printf("RECALL = %f, DEPTH = %d, TREE SIZE = %d\n", params.recall, params.depth, params.treeSize);

        generator.scs = cc->sc;
        generator.scSize = s + 1;

        train(cc->sc + s, posSet, negSet, &generator, &params);

        cc->ssize++;

        {
            char filePath[256];
            sprintf(filePath, "model/cascade_%d.dat", s);
            save(cc, filePath);
        }

        {
            char command[256];
            sprintf(command, "mv classifier.txt log/classifier_%d.txt", s);
            ret = system(command);
        }

        printf("---------------------------------------------\n");
    }

    fclose(generator.fin);

    release(&posSet);
    release(&negSet);

    return 0;
}


int predict(QTObjectDetector *cc, uint32_t *iImg, int iStride, float &score){
    int x0 = -1;
    int x1 = cc->WINW - 1;
    int y0 = -iStride;
    int y1 = (cc->WINH - 1)* iStride;

    score = 0;

    for(int i = 0; i < cc->ssize; i++){
        float t;
        if(predict(cc->sc + i, iImg, iStride, t) == 0)
            return 0;

        score = t;
    }

    return 1;
}


void refine_samples(QTObjectDetector *cc, SampleSet *set, int flag){
    int ssize = set->ssize;

    printf("refine sample %d, ", ssize);

    if(flag == 0){
        for(int i = 0; i < ssize; i++){
            float score = 0;
            Sample *sample = set->samples[i];

            predict(cc->sc, cc->ssize, sample->iImg, sample->istride, sample->score);
        }
    }
    else {
        for(int i = 0; i < ssize; i++){
            float score = 0;
            Sample *sample = set->samples[i];

            if(predict(cc->sc, cc->ssize, sample->iImg, sample->istride, sample->score) == 0){
                QT_SWAP(set->samples[i], set->samples[ssize - 1], Sample*);

                ssize --;
                i--;
            }
        }
    }

    set->ssize = ssize;

    printf("%d\n", set->ssize);
}



void init_detect_factor(QTObjectDetector *cc, float startScale, float endScale, float offset, int layer){
    cc->startScale = startScale;
    cc->endScale = endScale;
    cc->layer = layer;
    cc->offsetFactor = offset;

    float stepFactor = powf(endScale / startScale, 1.0f / layer);
}


#define MERGE_RECT



int calc_overlapping_area(QTRect &rect1, QTRect &rect2){
    int cx1 = rect1.x + rect1.width / 2;
    int cy1 = rect1.y + rect1.height / 2;
    int cx2 = rect2.x + rect2.width / 2;
    int cy2 = rect2.y + rect2.height / 2;

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < rect1.width / 2 + rect2.width/2 && abs(cy1 - cy2) < rect1.height / 2 + rect2.height / 2){
        x0 = QT_MAX(rect1.x , rect2.x);
        x1 = QT_MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        y0 = QT_MAX(rect1.y, rect2.y);
        y1 = QT_MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
    }
    else {
        return 0;
    }

    return (y1 - y0 + 1) * (x1 - x0 + 1);
}


int merge_rects(QTRect *rects, float *confs, int size){
    if(size < 2) return size;

    uint8_t *flags = new uint8_t[size];

    memset(flags, 0, sizeof(uint8_t) * size);

    for(int i = 0; i < size; i++){
        if(flags[i] == 1)
            continue;

        float area0 = 1.0f / (rects[i].width * rects[i].height);

        for(int j = i + 1; j < size; j++){
            if(flags[j] == 1) continue;

            float area1 = 1.0f / (rects[j].width * rects[j].height);

            int overlap = calc_overlapping_area(rects[i], rects[j]);

            if(overlap * area1 > 0.6f || overlap * area0 > 0.6f){
                if(confs[i] > confs[j])
                    flags[j] = 1;
                else
                    flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) {
            continue;
        }

        flags[i] = flags[size - 1];

        rects[i] = rects[size - 1];
        confs[i] = confs[size - 1];

        i --;
        size --;
    }

    delete []flags;
    flags = NULL;

    return size;
}


#define FIX_Q 14

int detect_one_scale(QTObjectDetector *cc, float scale, uint32_t *iImg, int width, int height, int stride, QTRect *resRect, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int count = 0;
    float score;

    int HALF_ONE = 1 << (FIX_Q - 1);
    int FIX_SCALE = scale * (1 << FIX_Q);

    int x0 = -1;
    int x1 = WINW - 1;
    int y0 = -stride;
    int y1 = (WINH - 1) * stride;

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint32_t *ptr = iImg + y * stride + x;

            if(predict(cc->sc, cc->ssize, iImg + y * stride + x, stride, score) == 1){
                resRect[count].x = (x * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].y = (y * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resRect[count].width = (WINW * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].height = (WINH * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resScores[count] = score;
                count++;
            }
        }
    }

#ifdef MERGE_RECT
    count = merge_rects(resRect, resScores, count);
#endif

    return count;
}


int calculate_max_size(int width, int height, float startScale, int winSize){
    int minwh = QT_MIN(width, height);

    assert(startScale < 1.0f);

    int size = minwh * startScale;
    float scale = (float)winSize / size;

    width ++;
    height ++;

    if(scale < 1)
        return width * height;

    return (width * scale + 0.5f) * (height * scale + 0.5f);
}


int detect(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride, QTRect **resRect, float **rscores){
    int WINW, WINH, capacity;
    float scale, stepFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;
    uint32_t *iImgBuf, *iImg;

    QTRect *rects;
    float *scores;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    scale = cc->startScale;
    stepFactor = powf(cc->endScale / cc->startScale, 1.0f / (cc->layer));

    capacity = calculate_max_size(width, height, scale, QT_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);
    iImgBuf = new uint32_t[capacity * 2]; assert(iImgBuf != NULL);

    const int BUFFER_SIZE = 1000;
    rects  = new QTRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);

    memset(rects, 0, sizeof(QTRect)  * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    minSide = QT_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        float scale2 = QT_MIN(WINW, WINH) / (minSide * scale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        QT_resize_gray_image(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        assert(dstw * dsth < 16777216);

        memset(iImgBuf, 0, sizeof(uint32_t) * (dstw + 1) * (dsth + 1));
        iImg = iImgBuf + dstw + 1 + 1;

        QT_integral_image(ptrDst, dstw, dsth, dsts, iImg, dstw + 1);

        count += detect_one_scale(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        scale *= stepFactor;
    }

    if(count > 0){
#ifdef MERGE_RECT
        count = merge_rects(rects, scores, count);
#endif

        *resRect = new QTRect[count]; assert(resRect != NULL);
        memcpy(*resRect, rects, sizeof(QTRect) * count);
        *rscores = new float[count]; assert(rscores != NULL);
        memcpy(*rscores, scores, sizeof(float) * count);
    }

    delete [] dImg;
    delete [] iImgBuf;
    delete [] rects;
    delete [] scores;

    return count;
}


int load(QTObjectDetector *cc, const char *filePath){
    if(cc == NULL)
        return 1;

    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 2;
    }

    int ret;

    char str[100];
    int versionEpoch, versionMajor, versionMinor;

    ret = fread(str, sizeof(char), 100, fin); assert(ret == 100);
    sscanf(str, "HANGZHOU QIANTU TECHNOLOGY FACE DETECTOR: %d.%d.%d", &versionEpoch, &versionMajor, &versionMinor);

    assert(versionEpoch == QT_VERSION_EPCH);
    assert(versionMajor == QT_VERSION_MAJOR);
    assert(versionMinor == QT_VERSION_MINOR);

    ret = fread(&cc->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINH, sizeof(int), 1, fin); assert(ret == 1);

    cc->sc = new StrongClassifier[cc->ssize]; assert(cc->sc != NULL);

    memset(cc->sc, 0, sizeof(StrongClassifier) * cc->ssize);

    for(int i = 0; i < cc->ssize; i++){
        ret = load(cc->sc + i, fin);
        if(ret != 0){
            printf("Load strong classifier error\n");
            fclose(fin);

            delete [] cc->sc;
            delete cc;

            return 2;
        }
    }

    fclose(fin);

    return 0;
}


int save(QTObjectDetector *cc, const char *filePath){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL)
        return 1;

    int ret;

    char str[100];

    sprintf(str, "HANGZHOU QIANTU TECHNOLOGY FACE DETECTOR: %d.%d.%d", QT_VERSION_EPCH, QT_VERSION_MAJOR, QT_VERSION_MINOR);

    ret = fwrite(str, sizeof(char), 100, fout);
    ret = fwrite(&cc->ssize, sizeof(int), 1, fout);
    ret = fwrite(&cc->WINW, sizeof(int), 1, fout);
    ret = fwrite(&cc->WINH, sizeof(int), 1, fout);

    for(int i = 0; i < cc->ssize; i++){
        ret = save(cc->sc + i, fout);
        if(ret != 0){
            printf("Save strong classifier error\n");
            fclose(fout);
            return 2;
        }
    }

    fclose(fout);

    return 0;
}


void release_data(QTObjectDetector *cc){
    if(cc->sc != NULL)
        return;

    for(int i = 0; i < cc->ssize; i++)
        release_data(cc->sc);

    delete [] cc->sc;
    cc->sc = NULL;
}


void release(QTObjectDetector **cc){
    if(*cc == NULL)
        return;

    release_data(*cc);
    delete cc;
    cc = NULL;
}
