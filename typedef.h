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


#ifndef _TYPE_DEFS_H_
#define _TYPE_DEFS_H_

#define QT_VERSION_EPCH 1
#define QT_VERSION_MAJOR 0
#define QT_VERSION_MINOR 0

#define QT_SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}
#define QT_MIN(i, j) ((i) > (j) ? (j) : (i))
#define QT_MAX(i, j) ((i) < (j) ? (j) : (i))
#define QT_ABS(a) ((a) < 0 ? (-a) : (a))


#define EPSILON 0.000001f
#define QT_PI 3.1415926535

#ifndef QT_FREE
#define QT_FREE(arr) \
{ \
    if(arr != NULL) \
        free(arr); \
    arr = NULL; \
}
#endif


#endif
