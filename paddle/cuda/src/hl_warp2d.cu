/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include "hl_base.h"
#include "paddle/utils/Logging.h"


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


__global__ void Warp2DForward(real* input,
                              real* flow, 
                              real* output,
                              const int channel,
                              const int height,
                              const int width) {
  int nthreads = height * width;
  CUDA_KERNEL_LOOP(index, nthreads) {

      const int w = index % width;
      const int h = (index / width) % height;
      // which channel 
      const int pix_num = height * width;
      // from the 
      float xx = float(w) + flow[index];
      float yy = float(h) + flow[index + pix_num];
      
      float x1 = float(floorf(xx));
      float x2 = x1 + 1.0;

      float y1 = float(floorf(yy));
      float y2 = y1 + 1.0;


      if (x1 < 0. || x1 > width-1. ||
          y1 < 0. || y1 > height-1.) {
          for (int cc = 0; cc<channel; cc++){
               int off =  cc * height * width;
               output[w + h * width + off] = 0;// index of a perticular r/g/b pixel in image:  h * width + w + (n*channels +  cc)*height*width
          }
      }
      else if (x2 > width-1. || y2 > height-1. ) {
        for (int cc = 0; cc < channel; cc ++){    
          int off = cc * height * width;
          // padding with boarder value
          float val = input[int(x1) + int(y1) * width + off];
          output[w + h * width + off] = val;
        }
      }
      else {
        for (int cc = 0; cc < channel; cc ++){    
          int off = cc * height * width;
          //bilinear interplate for the new value 
          float val_up, val_down, val;
          val_up = (x2-xx) * input[int(x1) + int(y1) * width + off] + 
                   (xx-x1) * input[int(x2) + int(y1) * width + off];
          val_down = (x2-xx) * input[int(x1) + int(y2) * width + off] + 
                     (xx-x1) * input[int(x2) + int(y2) * width + off];// I_in_x_y2 /= (x2-x1);  
          val = (y2-yy) * val_up + (yy-y1) * val_down; //I_in_xy /= (y2-y1);
          output[w + h * width + off] =  val;
        }           
      }
  }
}


void hl_warp2d_forward(real *input,
                       real *flow, 
                       real *output,
                       const int channel,
                       const int height,
                       const int width) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(flow);
  const int threads = 512;
  const int blocks = DIVUP(height * width, threads);

  Warp2DForward<<<blocks, threads, 0, STREAM_DEFAULT>>>(
        input, flow, output, channel, height, width);
  CHECK_SYNC("hl_warp2d_forward failed");
}

