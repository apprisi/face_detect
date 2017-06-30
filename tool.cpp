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


#include "tool.h"


int read_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fscanf(fin, "%s\n", line) != EOF){
        fileList.push_back(line);
    }

    fclose(fin);

    return 0;
}


void split_file_path(const char* filePath, char *rootDir, char *fileName, char *ext)
{
    int len = strlen(filePath);
    int idx = len - 1, idx2 = 0;

    while(idx >= 0){
        if(filePath[idx] == '.')
            break;
        idx--;
    }

    if(idx >= 0){
        strcpy(ext, filePath + idx + 1);
        ext[len - idx] = '\0';
    }
    else {
        ext[0] = '\0';
        idx = len - 1;
    }

    idx2 = idx;
    while(idx2 >= 0){
#ifdef WIN32
        if(filePath[idx2] == '\\')
#else
        if(filePath[idx2] == '/')
#endif
            break;

        idx2 --;
    }

    if(idx2 > 0){
        strncpy(rootDir, filePath, idx2);
        rootDir[idx2] = '\0';
    }
    else{
        rootDir[0] = '.';
        rootDir[1] = '\0';
    }

    strncpy(fileName, filePath + idx2 + 1, idx - idx2 - 1);
    fileName[idx - idx2 - 1] = '\0';
}


void QT_integral_image(uint8_t *img, int width, int height, int stride, uint32_t *iImg, int istride){

    uint32_t *ptrLine1 = iImg;
    uint32_t *ptrLine2 = iImg + istride;

    for(int x = 0; x < width; x++)
        iImg[x] = iImg[x - 1] + img[x];

    img += stride;

    for(int y = 1; y < height; y ++){
        uint32_t sum = 0;

        for(int x = 0; x < width; x++){
            sum += img[x];
            ptrLine2[x] = ptrLine1[x] + sum;
        }

        img += stride;
        ptrLine1 += istride;
        ptrLine2 += istride;
    }
}




void update_weights(double *weights, int size){
    double sum = 0.0;

    for(int i = 0; i < size; i++)
        sum += weights[i];

    sum = 1.0 / sum;
    for(int i = 0; i < size; i++){
        weights[i] *= sum;
    }

}


#define QT_FIX_POINT_Q 14

void QT_resize_gray_image(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *xtable = NULL;

    uint16_t FIX_0_5 = 1 << (QT_FIX_POINT_Q - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    xtable = new uint16_t[dstw * 2];

    for(int x = 0; x < dstw; x++){
        float xc = x * scalex;

        if(xc < 0) xc = 0;
        if(xc >= srcw - 1) xc = srcw - 1.01f;

        int x0 = int(xc);

        xtable[x * 2] = x0;
		xtable[x * 2 + 1] = (1 << QT_FIX_POINT_Q) - (xc - x0) * (1 << QT_FIX_POINT_Q);
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0;
        uint16_t y0, y1;
        uint16_t *ptrTab = xtable;

        yc = y * scaley;

        if(yc < 0) yc = 0;
        if(yc >= srch - 1) yc = srch - 1.01f;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy0 = (1 << QT_FIX_POINT_Q) - uint16_t((yc - y0) * (1 << QT_FIX_POINT_Q));

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0;
			int vy0, vy1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[0] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //2
            x0 = ptrTab[2], x1 = x0 + 1;
            wx0 = ptrTab[3];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[1] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //3
            x0 = ptrTab[4], x1 = x0 + 1;
            wx0 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[2] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //4
            x0 = ptrTab[6], x1 = x0 + 1;
            wx0 = ptrTab[7];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[3] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            ptrDst += 4;
            ptrTab += 8;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, vy0, vy1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            dst[y * dsts + x] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            ptrTab += 2;
        }

        dId += dsts;
    }


    delete [] xtable;
}



void QT_affine_image(uint8_t *src, int srcw, int srch, int srcs, float angle, float scale, cv::Point2f &center,
        uint8_t *dst, int dstw, int dsth, int dsts){

    int FIX_ONE = 1 << QT_FIX_POINT_Q;
    int FIX_0_5 = FIX_ONE >> 1;

    float sina = sin(-angle) / scale;
    float cosa = cos(-angle) / scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);

    float cx = (float)dstw / 2;
    float cy = (float)dsth / 2;

    int fcx = center.x * FIX_ONE;
    int fcy = center.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - cy);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - cx);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

    id = 0;
    for(int y = 0; y < dsth; y++){
        int idx = y << 1;

        int ys = ytable[idx]    ;
        int yc = ytable[idx + 1];

        for(int x = 0; x < dstw; x++){
            idx = x << 1;

            int xs = xtable[idx];
            int xc = xtable[idx + 1];

            int fx =  xc + ys;
            int fy = -xs + yc;

            int x0 = fx >> QT_FIX_POINT_Q;
            int y0 = fy >> QT_FIX_POINT_Q;

            int wx = fx - (x0 << QT_FIX_POINT_Q);
            int wy = fy - (y0 << QT_FIX_POINT_Q);

            if(x0 < 0 || x0 >= srcw || y0 < 0 || y0 >= srch)
                continue;

            assert(wx <= FIX_ONE && wy <= FIX_ONE);

            uint8_t *ptr1 = src + y0 * srcs + x0;
            uint8_t *ptr2 = ptr1 + srcs;

            uint8_t value0 = ((ptr1[0] << QT_FIX_POINT_Q) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> QT_FIX_POINT_Q;
            uint8_t value1 = ((ptr2[0] << QT_FIX_POINT_Q) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> QT_FIX_POINT_Q;

            dst[id + x] = ((value0 << QT_FIX_POINT_Q) + (value1 - value0) * wy + FIX_0_5) >> QT_FIX_POINT_Q;
        }

        id += dsts;
    }

    delete [] xtable;
}


void transform_image(uint8_t *img, int width, int height, int stride, uint8_t *dImg){
    static cv::RNG rng(cv::getTickCount());

    float angle = rng.uniform(-QT_PI, QT_PI);
    float scale = rng.uniform(0.7, 1.3);

    assert(stride * height < 4096 * 4096);

    cv::Point2f center(width >> 1, height >> 1);

    memset(dImg, 0, sizeof(uint8_t) * height * stride);

    QT_affine_image(img, width, height, stride, angle, scale, center, dImg, width, height, stride);

    memcpy(img, dImg, height * stride);

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        float sum = 0;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                sum += data[x];

            data += stride;
        }

        sum /= (width * height);
        sum ++;

        data = img;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                data[x] = ((data[x] - sum) / (data[x] + sum) + 1.0f) * 0.5 * 255;

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-8, 8);
            }

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0)
        vertical_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0)
        horizontal_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::GaussianBlur(sImg, blur, cv::Size(5, 5), 0, 0);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }


    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::equalizeHist(sImg, blur);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }
}



void transform_image(cv::Mat &img, int WINW){
    static cv::RNG rng(cv::getTickCount());

    int w = rng.uniform(3 * WINW, 10 * WINW);
    int h = w * img.rows / img.cols;

    assert(w > 0 && h > 0);
    float angle = rng.uniform(-180.0f, 180.0f);

    cv::Mat affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols >> 1, img.rows >> 1), angle, float(w) / img.cols);

    cv::warpAffine(img, img, affineMat, cv::Size(w, h));

    if(rng.uniform(0, 8) == 0){
        cv::equalizeHist(img, img);
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img.data;

        float sum = 0;
        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++)
                sum += data[x];

            data += img.step;
        }

        sum /= (img.cols * img.rows);
        sum += 1;

        data = img.data;
        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++)
                data[x] = ((data[x] - sum) / (data[x] + sum) + 1.0f) * 0.5 * 255;

            data += img.step;
        }
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img.data;

        for(int y = 0; y < img.rows; y++){
            for(int x = 0; x < img.cols; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-16, 16);
            }

            data += img.step;
        }
    }
}


void vertical_mirror(uint8_t *img, int width, int height, int stride)
{
    int cy = height / 2;

    for(int y = 0; y < cy; y++){
        uint8_t *ptr1 = img + y * stride;
        uint8_t *ptr2 = img + (height - y - 1) * stride;

        for(int x = 0; x < width; x++)
            QT_SWAP(ptr1[x], ptr2[x], uint8_t);
    }
}


void horizontal_mirror(uint8_t *img, int width, int height, int stride){
    int cx = width / 2;
    uint8_t *ptr = img;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < cx; x++)
            QT_SWAP(ptr[x], ptr[width - x - 1], uint8_t);
        ptr += stride;
    }
}


#define LT(a, b) (a < b)

//come from opencv2/core/internal.hpp CV_IMPLMENT_QSORT
void quick_sort_float( float *array, int total_num)
{
    int isort_thresh = 7;
    int sp = 0;

    struct
    {
        float *lb;
        float *ub;
    }
    stack[48];

    if( total_num <= 1 )
        return;

    stack[0].lb = array;
    stack[0].ub = array + (total_num - 1);

    while( sp >= 0 )
    {
        float* left = stack[sp].lb;
        float* right = stack[sp--].ub;

        for(;;)
        {
            int i, n = (int)(right - left) + 1, m;
            float* ptr;
            float* ptr2;

            if( n <= isort_thresh )
            {
insert_sort_quick_sort_float:
                for( ptr = left + 1; ptr <= right; ptr++ )
                {
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1]); ptr2--)
                        QT_SWAP( ptr2[0], ptr2[-1], float);
                }
                break;
            }
            else
            {
                float *left0, *left1;
                float *right0, *right1;
                float *pivot;
                float *a, *b, *c;

                int swap_cnt = 0;

                left0 = left;
                right0 = right;
                pivot = left + (n >> 1);

                if( n > 40 )
                {
                    int d = n >> 3;
                    a = left, b = left + d, c = left + 2*d;
                    left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a)) : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));

                    a = pivot - d, b = pivot, c = pivot + d;
                    pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a)) : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));

                    a = right - 2*d, b = right - d, c = right;
                    right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a)) : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));
                }

                a = left, b = pivot, c = right;
                pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a)) : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));
                if( pivot != left0 ){
                    QT_SWAP( *pivot, *left0, float );
                    pivot = left0;
                }
                left = left1 = left0 + 1;
                right = right1 = right0;

                for(;;)
                {
                    while( left <= right && !LT(*pivot, *left) )
                    {
                        if( !LT(*left, *pivot) )
                        {
                            if( left > left1 )
                                QT_SWAP( *left1, *left, float );
                            swap_cnt = 1;
                            left1++;
                        }
                        left++;
                    }

                    while( left <= right && !LT(*right, *pivot) )
                    {
                        if( !LT(*pivot, *right) )
                        {
                            if( right < right1 )
                                QT_SWAP( *right1, *right, float);
                            swap_cnt = 1;
                            right1--;
                        }
                        right--;
                    }

                    if( left > right )
                        break;
                    QT_SWAP( *left, *right, float );
                    swap_cnt = 1;
                    left++;
                    right--;
                }

                if( swap_cnt == 0 )
                {
                    left = left0, right = right0;
                    goto insert_sort_quick_sort_float;
                }

                n = QT_MIN( (int)(left1 - left0), (int)(left - left1) );
                for( i = 0; i < n; i++ )
                    QT_SWAP( left0[i], left[i-n], float );

                n = QT_MIN( (int)(right0 - right1), (int)(right1 - right) );
                for( i = 0; i < n; i++ )
                    QT_SWAP( left[i], right0[i-n+1], float );
                n = (int)(left - left1);
                m = (int)(right1 - right);
                if( n > 1 )
                {
                    if( m > 1 )
                    {
                        if( n > m )
                        {
                            stack[++sp].lb = left0;
                            stack[sp].ub = left0 + n - 1;
                            left = right0 - m + 1, right = right0;
                        }
                        else
                        {
                            stack[++sp].lb = right0 - m + 1;
                            stack[sp].ub = right0;
                            left = left0, right = left0 + n - 1;
                        }
                    }
                    else
                        left = left0, right = left0 + n - 1;
                }
                else if( m > 1 )
                    left = right0 - m + 1, right = right0;
                else
                    break;
            }
        }
    }
}


void sleep(uint64_t milisecond){
    clock_t st = clock();
    clock_t et = st + milisecond * CLOCKS_PER_SEC / 1000;

    while(clock() < et){}
}
