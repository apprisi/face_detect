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


#include "classifier.h"


int generate_negative_images(const char *listFile, const char *outfile){
    FILE *fin = fopen(outfile, "rb");
    if(fin != NULL){
        fclose(fin);
        return 0;
    }

    std::vector<std::string> imgList;
    read_list(listFile, imgList);

    int size = imgList.size();

    printf("GENERATE NEGATIVE IMAGES %ld\n", imgList.size());
    FILE *fout = fopen(outfile, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", outfile);
        return 1;
    }

    char rootDir[256], fileName[256], ext[30];
    int ret;

    ret = fwrite(&size, sizeof(int), 1, fout), assert(ret == 1);

    for(int i = 0; i < size; i++){
        const char *imgPath = imgList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 0);

        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        split_file_path(imgPath, rootDir, fileName, ext);

        ret = fwrite(fileName, sizeof(char), 255, fout); assert(ret == 255);

        if(img.cols > img.rows && img.rows > 720)
            cv::resize(img, img, cv::Size(img.cols * 720 / img.rows, 720));
        else if(img.rows > img.cols && img.cols > 720)
            cv::resize(img, img, cv::Size(720, 720 * img.rows / img.cols));

        if(img.rows * img.cols > 1024 * 1024){
            float scale = 1024.0f / img.rows;
            scale = QT_MIN(1024.0f / img.cols, scale);
            cv::resize(img, img, cv::Size(scale * img.cols, scale * img.rows));
        }

        assert(img.cols * img.rows <= 1024 * 1024);

        ret = fwrite(&img.rows, sizeof(int), 1, fout); assert(ret == 1);
        ret = fwrite(&img.cols, sizeof(int), 1, fout); assert(ret == 1);

        if(img.cols == img.step)
            ret = fwrite(img.data, sizeof(uint8_t), img.rows * img.cols, fout);
        else
            for(int y = 0; y < img.rows; y++)
                ret = fwrite(img.data + y * img.step, sizeof(uint8_t), img.cols, fout);
        if(i % 10000 == 0)
            printf("%d ", i), fflush(stdout);
    }

    printf("\n");
    fclose(fout);

    printf("IMAGE SIZE: %d\n", size);
    return 0;
}


void load_images(FILE *fin, uint8_t **imgs, int *widths, int *heights, char **fileNames, int size, int tflag){
    int ret = 0;

    uint8_t *buffer = new uint8_t[4096 * 4096];

    for(int j = 0; j < size; j++){
        int width, height;

        ret = fread(fileNames[j], sizeof(char), 255, fin); assert(ret == 255);
        ret = fread(heights + j, sizeof(int), 1, fin); assert(ret == 1);
        ret = fread(widths + j, sizeof(int), 1, fin); assert(ret == 1);

        ret = fread(imgs[j], sizeof(uint8_t), widths[j] * heights[j], fin); assert(ret == widths[j] * heights[j]);

        if(tflag)
            transform_image(imgs[j], widths[j], heights[j], widths[j], buffer);

        /*
        cv::Mat sImg(heights[j], widths[j], CV_8UC1, imgs[j]);
        cv::imshow("sImg", sImg);
        cv::waitKey();
        //*/
    }

    delete [] buffer;
}


int detect_image(uint8_t *img, int width, int height, int stride, char *fileName,
        NegGenerator *generator, uint8_t *buffer, uint32_t *intImg, SampleSet *set, int maxSize){
    int count = 0;
    float scale = 1.0f;

    int WINW = set->winw;
    int WINH = set->winh;

    int dx = generator->dx;
    int dy = generator->dy;

    uint32_t *iImg;
    int istride;

    for(int l = 0; l < 20; l++){
        int w, h, s;

        if(width > height){
            h = WINH * scale;
            w = h * width / height;
            w = QT_MAX(w, WINW);

        }
        else {
            w = WINW * scale;
            h = w * height / width;
            h = QT_MAX(h, WINH);
        }

        if(w * h > 4096 * 4096) break;

        s = w;
        QT_resize_gray_image(img, width, height, stride, buffer, w, h, w);

        memset(intImg, 0, sizeof(uint32_t) * (w + 1) * (h + 1));

        iImg = intImg + w + 1 + 1;
        istride = w + 1;

        QT_integral_image(buffer, w, h, w, iImg, istride);

        for(int y = 0; y <= h - WINH; y += dy){
            for(int x = 0; x <= w - WINW; x += dx){
                uint32_t *ptr = iImg + y * istride + x;
                float score;

                if(predict(generator->scs, generator->scSize, ptr, istride, score) == 1){
                    char filePath[256];
                    sprintf(filePath, "%s_%d", fileName, count++);

                    Sample *sample = new Sample;
                    memset(sample, 0, sizeof(Sample));

                    create_sample(sample, buffer + y * w + x, WINW, WINH, w, filePath);
                    sample->score = score;

                    add_sample_capacity_unchange(set, sample);
                }
            }
        }

        if(count > maxSize)
            break;

        scale *= 1.13f;
    }

    return count;
}


void generate_negative_samples(SampleSet *negSet, NegGenerator *generator, int needSize){
    if(needSize <= 0)
        return;

    int WINW = negSet->winw;
    int WINH = negSet->winh;

    const int BUF_SIZE = 1000;

    uint8_t **imgs;
    char **fileNames;
    int *widths, *heights;

    imgs = new uint8_t*[BUF_SIZE];
    widths  = new int[BUF_SIZE];
    heights = new int[BUF_SIZE];
    fileNames = new char*[BUF_SIZE];

    for(int i = 0; i < BUF_SIZE; i++){
        fileNames[i] = new char[256];
        imgs[i] = new uint8_t[1024 * 1024];
    }

    int threadNum = omp_get_num_procs() - 1;

    SampleSet *sets = new SampleSet[threadNum];
    uint8_t **sImgs = new uint8_t*[threadNum];
    uint32_t **intImgs = new uint32_t*[threadNum];

    memset(sets, 0, sizeof(SampleSet) * threadNum);

    for(int i = 0; i < threadNum; i++){
        sImgs[i] = new uint8_t[4096 * 4096];
        intImgs[i] = new uint32_t[4097 * 4097];

        sets[i].winw = WINW;
        sets[i].winh = WINH;
        sets[i].ssize = 0;

        reserve(sets + i, (needSize / threadNum + 1) * 2.0f);
    }

    int id = generator->id;
    int ret;
    int ssize;

    //printf("%d %d %d\n", generator->isize, generator->id, generator->fsize);
    while(1){
        printf("%d ", id);
        int ei = QT_MIN(BUF_SIZE, generator->isize - id);

        //printf("Load images %d, ", ei);fflush(stdout);
        load_images(generator->fin, imgs, widths, heights, fileNames, ei, generator->tflag);

        //printf("sample, "); fflush(stdout);
#pragma omp parallel for num_threads(threadNum)
        for(int i = 0; i < ei; i++){
            int tId = omp_get_thread_num();
            assert(tId < threadNum);

            uint8_t *img = imgs[i];

            int width = widths[i];
            int height = heights[i];

            char *fileName = fileNames[i];

            uint8_t *sImg = sImgs[tId];
            uint32_t *intImg = intImgs[tId];
            SampleSet *set = sets + tId;

            int count = detect_image(img, width, height, width, fileName, generator, sImg, intImg, set, generator->maxCount);
        }

        ssize = 0;
        for(int i = 0; i < threadNum; i++)
            ssize += sets[i].ssize;

        id += BUF_SIZE;

        if(id >= generator->isize){
            id = 0;
            fclose(generator->fin);
            generator->fin = fopen(NEG_IMAGES_FILE, "rb"); assert(generator->fin != NULL);
            ret = fread(&generator->isize, sizeof(int), 1, generator->fin);
            assert(ret == 1);
        }

        printf("%d, ", ssize);fflush(stdout);

        if(ssize > needSize)
            break;
    }
    printf("\n");

    generator->id = id;

    ssize += negSet->ssize;
    reserve(negSet, ssize);

    for(int i = 0; i < threadNum; i++){
        memcpy(negSet->samples + negSet->ssize, sets[i].samples, sizeof(Sample*) * sets[i].ssize);
        negSet->ssize += sets[i].ssize;

        delete [] sets[i].samples;

        sets[i].samples = NULL;
        sets[i].ssize = 0;
        sets[i].capacity = 0;

        delete [] sImgs[i];
        delete [] intImgs[i];
    }

    delete [] sets;
    delete [] sImgs;
    delete [] intImgs;

    for(int i = 0; i < BUF_SIZE; i++){
        delete [] imgs[i];
        delete [] fileNames[i];
    }

    delete [] imgs;
    delete [] fileNames;
    delete [] widths;
    delete [] heights;
}


void refine_samples(SampleSet *set, float thresh){
    for(int i = 0; i < set->ssize; i++){
        if(set->samples[i]->score > thresh)
            continue;

        QT_SWAP(set->samples[i], set->samples[set->ssize - 1], Sample*);
        release(&set->samples[set->ssize - 1]);
        i--;
        set->ssize --;
    }
}


static void print_recall_and_precision(SampleSet *posSet, SampleSet *negSet, float thresh){
    float TP = 0, FP = 0;
    float FN = 0, TN = 0;

    for(int i = 0; i < posSet->ssize; i++){
        Sample* sample = posSet->samples[i];

        if(sample->score <= thresh)
            FN++;
        else
            TP ++;
    }

    for(int i = 0; i < negSet->ssize; i++){
        Sample* sample = negSet->samples[i];
        if(sample->score <= thresh)
            TN ++;
        else
            FP ++;
    }

    TP /= posSet->ssize;
    FN /= posSet->ssize;

    TN /= negSet->ssize;
    FP /= negSet->ssize;

    printf("recall: %f, precision: %f, thresh: %f\n", TP, TP / (TP + FP), thresh);
}


void train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, NegGenerator *generator, TrainParams *params){
    sc->depth = params->depth;
    sc->treeSize = 0;
    sc->capacity = params->treeSize;
    sc->trees = new Tree*[params->treeSize];
    sc->threshes = new float[params->treeSize];

    int nlSize = (1 << sc->depth) - 1;

    FILE *fout = fopen("classifier.txt", "w");

    for(int i = 0; i < posSet->ssize; i++)
        posSet->samples[i]->score = 0;

    for(int i = 0; i < negSet->ssize; i++)
        negSet->samples[i]->score = 0;

    for(int i = 0; i < params->treeSize; i++){
        printf("classifier: %d\n", i);
        Tree *tree = new Node[nlSize];
        memset(tree, 0, sizeof(Node) * nlSize);

        int needSize = generator->npRate * posSet->ssize - negSet->ssize;

        printf("GENERATE NEGATIVE SAMPLES %d\n", needSize);
        generate_negative_samples(negSet, generator, needSize);

        assert(negSet->ssize > 0.9 * posSet->ssize);

        sc->threshes[i] = train(tree, sc->depth, posSet, negSet, params->recall);

        if(params->flag){
            refine_samples(negSet, sc->threshes[i]);
            printf("posSize: %d, negSize: %d, thresh: %f\n", posSet->ssize, negSet->ssize, sc->threshes[i]);
        }
        else{
            print_recall_and_precision(posSet, negSet, sc->threshes[i]);
        }

        sc->trees[i] = tree;
        sc->treeSize++;

        print_tree(tree, sc->depth, fout);

        printf("\n");
    }

    fclose(fout);

    if(params->flag == 0){
        refine_samples(negSet, sc->threshes[sc->treeSize - 1]);
        write_images(negSet, "log/neg", 1);
    }

    release_data(negSet);
}


int predict(StrongClassifier *sc, uint32_t *intImg, int istride, float &score){
    int i = 0;

    score = 0;

    for(i = 0; i < sc->treeSize; i++){
        score += predict(sc->trees[i], sc->depth, intImg, istride);
        if(score <= sc->threshes[i])
            return 0;
    }

    return 1;
}


int predict(StrongClassifier *sc, int scSize, uint32_t *intImg, int istride, float &score){
    score = 0;
    for(int i = 0; i < scSize; i++){
        StrongClassifier *ptrSC = sc + i;

        Tree **ptrTrees = ptrSC->trees;
        int depth = ptrSC->depth;
        int treeSize = ptrSC->treeSize;

        float sscore = 0;
        int j;

        for(j = 0; j < treeSize; j++){
            sscore += predict(ptrTrees[j], depth, intImg, istride);
            if(sscore <= ptrSC->threshes[j]){
                return 0;
            };
        }

        score = sscore;
    }

    return 1;
}


int save(StrongClassifier *sc, FILE *fout){
    if(fout == NULL || sc == NULL)
        return 1;

    int ret;

    ret = fwrite(&sc->treeSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&sc->depth, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(sc->threshes, sizeof(float), sc->treeSize, fout); assert(ret == sc->treeSize);

    for(int i = 0; i < sc->treeSize; i++)
        save(sc->trees[i], sc->depth, fout);

    return 0;
}


int load(StrongClassifier *sc, FILE *fin){
    if(fin == NULL || sc == NULL){
        return 1;
    }

    int ret;
    int nlSize;

    ret = fread(&sc->treeSize, sizeof(int), 1, fin); assert(ret == 1);

    sc->trees = new Tree*[sc->treeSize];
    sc->threshes = new float[sc->treeSize];

    ret = fread(&sc->depth, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(sc->threshes, sizeof(float), sc->treeSize, fin); assert(ret == sc->treeSize);


    nlSize = (1 << sc->depth) - 1;

    sc->capacity  = sc->treeSize;

    for(int i = 0; i < sc->treeSize; i++){
        sc->trees[i] = new Node[nlSize];
        memset(sc->trees[i], 0, sizeof(Node) * nlSize);
        load(sc->trees[i], sc->depth, fin);
    }

    return 0;
}


void release(StrongClassifier **sc){
    if(sc == NULL)
        return;

    release_data(*sc);

    delete *sc;
    *sc = NULL;
}


void release_data(StrongClassifier *sc){
    if(sc != NULL){
        if(sc->trees != NULL){
            for(int i = 0; i < sc->treeSize; i++){
                if(sc->trees[i] != NULL)
                    delete [] sc->trees[i];
                sc->trees[i] = NULL;
            }

            delete [] sc->trees;
        }

        sc->trees = NULL;

        if(sc->threshes != NULL)
            delete [] sc->threshes;

        sc->threshes = NULL;
    }

    sc->capacity = 0;
    sc->treeSize = 0;
}
