# face detect 
This project is an implementation of a face detector which is researched by Hanzhou Qiantu Technology Copyright.

## Attention
Before you download, copy, or view sources file of this project, you must obeyed the following regulations,

    1) Using for personal applications or study are freedom.
    2) Using for bussiness must ask us for agreements.

## Source code
This project built on ubuntu 14.04, depending on opencv-2.4.x. If your environment are linux and you opencv was
installded under /usr/local. You can use make in the root directory to build this project. There will be four excutable
file in directory bin.

    train  training model program, usage: ./train [pos sample list] [negative sample list] [out model]
    idetect detect images, usage: ./idetect [model] [image list file]
    vdetect detect videos, usage: ./vdetect [model] [video]

    model.dat the already trained model
If you only keep the detect source code, it's purge c++ code.

## Method
The algorithm in this project to detect face is real adaboost plus haar like features.

## Contribution
If you like this project and want to help us, you may joint or denote us.

## Contact 
* email: <business@qiantuai.com>
* website: [http://www.qiantuai.com](http://www.qiantuai.com)
* qq群： 535810799

If you want more help, you may send email to us. 

## Reference 
* Robust real-time face detection. PAUL VIOLA, MICHAEIL J. JONES
* A fast and accurate constrained face face detector. Shengcai Liao, Anil K. Jain, Stan Z. Li 
* Joint cascade face detection and alignment. Dong Chen, Shaoqing Ren, Yichen Wei, Xudong Chao, Jian Sun.

