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

#ifndef HL_WARP2D_H_
#define HL_WARP2D_H_

#include "hl_base.h"

/**
 * @brief   warp a 2d feature with optical flow.
 *
 * @param[in]   input          the 2d input tensor (C x H x W)
 * @param[in]   flow           optical flow to the 2d image.
 * @param[out]  output         top k index.
 * @param[in]   channels       channel number.
 * @param[in]   lds            height of the tensor.
 * @param[in]   dim            width of the tensor
 *
 */
extern void hl_warp2d_forward(real *input,
                              real *flow, 
                              real *output,
                              const int channel,
                              const int height,
                              const int width);

#endif  // HL_WARP2D_H_
