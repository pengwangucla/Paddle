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
#include <stdio.h>


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
      float xx = float(w) + flow[index] * float(width);
      float yy = float(h) + flow[index + pix_num] * float(height);
      
      float x1 = float(floorf(xx));
      float x2 = x1 + 1.0;

      float y1 = float(floorf(yy));
      float y2 = y1 + 1.0;


      if (x1 < 0. || x1 > width-1. ||
          y1 < 0. || y1 > height-1.) {
          // index of a perticular r/g/b pixel in image:
          //  h * width + w + (n*channels +  cc)*height*width
          for (int cc = 0; cc<channel; cc++){
               int off =  cc * height * width;
               output[w + h * width + off] = 0;
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
          // I_in_x_y2 /= (x2-x1);  
          val_down = (x2-xx) * input[int(x1) + int(y2) * width + off] + 
                     (xx-x1) * input[int(x2) + int(y2) * width + off];
          //I_in_xy /= (y2-y1);
          val = (y2-yy) * val_up + (yy-y1) * val_down; 
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


__device__ inline void Angle2Matrix(const real* ang,
                                    real* R) {
    real angle = sqrtf(ang[0] * ang[0] + 
                   ang[1] * ang[1] + 
                   ang[2] * ang[2]);

    if( angle > real(1e-6) )
    {
      real c = cosf(angle);
      real s = sinf(angle);
      real u[3] = {ang[0]/angle, ang[1]/angle, ang[2]/angle};

      R[0] = c+u[0]*u[0]*(1-c);      
      R[3] = u[1]*u[0]*(1-c)+u[2]*s; 
      R[6] = u[2]*u[0]*(1-c)-u[1]*s; 

      R[1] = u[0]*u[1]*(1-c)-u[2]*s; 
      R[4] = c+u[1]*u[1]*(1-c);      
      R[7] = u[2]*u[1]*(1-c)+u[0]*s; 

      R[2] = u[0]*u[2]*(1-c)+u[1]*s;
      R[5] = u[1]*u[2]*(1-c)-u[0]*s;
      R[8] = c+u[2]*u[2]*(1-c);
    }
    else
    {
      R[0] = 1; R[3] = 0; R[6] = 0;
      R[1] = 0; R[4] = 1; R[7] = 0;
      R[2] = 0; R[5] = 0; R[8] = 1;
    }
}

__device__ void MatMultiply(real* A, size_t row_A, size_t col_A,
                            real* B, size_t row_B, size_t col_B,
                            real* C) {
  for(int i = 0; i < row_A; i ++)
    for(int m = 0; m < col_B; m ++) {
      C[i * col_B + m] = 0.0f;
      for(int j = 0; j < col_A; j ++) {
          C[i * col_B + m] += A[i * col_A + j] * B[j * col_B + m];
      }
    }
}


// trans: [fx, fy, ux, uy, a1, a2, a3, t1, t2, t3]
__global__ void Depth2FlowForward(real* depth,
                                real* trans, 
                                real* flow,
                                const int batch_size,
                                const int height,
                                const int width) {

  int nthreads = height * width * batch_size;
  CUDA_KERNEL_LOOP(index, nthreads) {

    // transfer depth to 3d 
    // default is set and transfer to normalized flow
    int x_i = index % width;
    int y_i = (index / width) % height;
    real x = (real(x_i) + 0.5f) / real(width);
    real y = (real(y_i) + 0.5f) / real(height);

    int batch_id = index / (height * width);
    int image_size = height * width;

    real* cur_trans = trans + 10 * batch_id;

    real* f = cur_trans;
    real* u = cur_trans + 2;
    real* r = cur_trans + 4;
    real* t = cur_trans + 7;

    real x_3d[3] = {0.0f, 0.0f, 0.0f};
    x_3d[0] = (x - u[0]) / f[0] * depth[index];
    x_3d[1] = (y - u[1]) / f[1] * depth[index];
    x_3d[2] = depth[index];

    real R[9];
    Angle2Matrix(r, R);

    // project 3d to the second image
    real x_tmp_3d[3] = {0.f, 0.f, 0.f};
    MatMultiply(R, 3, 3, x_3d, 3, 1, x_tmp_3d);
    // for(int i = 0; i < 3; i ++)
    //   for(int j = 0; j < 3; j ++)
    //     x_tmp_3d[i] += R[i*3 + j] * x_3d[j];

    for(int i = 0; i < 3; i ++) x_3d[i] = x_tmp_3d[i];
    for(int i = 0; i < 3; i ++) x_3d[i] += t[i];

    // calculate the flow
    real x2 = x_3d[2] == 0.f ? x : ((x_3d[0] / x_3d[2] * f[0]) + u[0]);
    real y2 = x_3d[2] == 0.f ? y : ((x_3d[1] / x_3d[2] * f[1]) + u[1]);

    real norm = sqrtf((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));
    flow[batch_id * 2 * image_size + y_i * width + x_i] = 
                  norm > 1.0f ? 0. : x2 - x;
    flow[(batch_id * 2 + 1) * image_size + y_i * width + x_i] = 
                  norm > 1.0f ? 0. : y2 - y;
  }
}



// trans: [fx, fy, ux, uy, a1, a2, a3, t1, t2, t3]
__global__ void Flow2DepthForward(real* flow,
                              real* trans, 
                              real* depth,
                              const int batch_size,
                              const int height,
                              const int width) {
    // project 3d to the second image
  int nthreads = height * width * batch_size;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int x_i = index % width;
    int y_i = (index / width) % height;
    real x = (real(x_i) + 0.5f) / real(width);
    real y = (real(y_i) + 0.5f) / real(height);

    int batch_id = index / (height * width);
    int image_size = height * width;

    real x2 = x + 
      flow[batch_id * 2 * image_size + y_i * width + x_i];
    real y2 = y + 
      flow[(batch_id * 2 + 1) * image_size + y_i * width + x_i];

    real* cur_trans = trans + 10 * batch_id;
    real* f = cur_trans;
    real* u = cur_trans + 2;
    real* r = cur_trans + 4;
    real* t = cur_trans + 7;

    real x_3d[3] = {0.0f, 0.0f, 1.0f};
    x_3d[0] = (x - u[0]) / f[0];
    x_3d[1] = (y - u[1]) / f[1];

    real x2_3d[2] = {0.0f, 0.0f};
    x2_3d[0] = (x2 - u[0]) / f[0];
    x2_3d[1] = (y2 - u[1]) / f[1];

    real R[9];
    Angle2Matrix(r, R);

    /* the function is from d1 * R * x_3d + t = x2_3d
                      and x_2d = d1 * K{-1} * x_3d 
                      and x_2d + flow_2d = x2_3d[2] * K{-1} * x2_3d
    solve d1 should use svd, but we sample each pair and get the mean
    results for approximation */

    real res[3];
    MatMultiply(R, 3, 3, x_3d, 3, 1, res);
    real depth_cur = 0.0f;
    real counter = 0.0f;

    real div = res[2] * x2_3d[1] - res[1];
    if( div != 0) {
      depth_cur += (t[1] - x2_3d[1] * t[2]) / div;
      counter += 1.0f;
    }

    div = res[2] * x2_3d[0] - res[0];
    if(div != 0) {
      depth_cur += (t[0] - x2_3d[0] * t[2]) / div;
      counter += 1.0f;
    }

    div = x2_3d[1] * res[0] - x2_3d[0] * res[1] ;
    if(div != 0) {
      depth_cur += (x2_3d[0] * t[1] - x2_3d[1] * t[0]) / div;
      counter += 1.0f;
    }

    depth_cur = depth_cur / max(counter, real(1e-6));
    depth[index] = depth_cur <= 0. ? 0.0f : depth_cur;
  }
}


// trans: [fx, fy, ux, uy, a1, a2, a3, t1, t2, t3]
void hl_trans_depth_flow_forward(real *input,
                       real *trans, 
                       real *output,
                       const int batch_size,
                       const int height,
                       const int width,
                       const bool depth_to_flow) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(trans);

  const int threads = 1024;
  const int blocks = DIVUP(height * width * batch_size, threads);
  
  if(depth_to_flow) {
    Depth2FlowForward<<<blocks, threads, 0, STREAM_DEFAULT>>>(
          input, trans, output, batch_size, height, width);
  }
  else {
    Flow2DepthForward<<<blocks, threads, 0, STREAM_DEFAULT>>>(
          input, trans, output, batch_size, height, width);
  }

  CHECK_SYNC("hl_trans_depth_flow_forward failed");
}