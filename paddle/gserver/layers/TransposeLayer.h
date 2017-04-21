/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "Layer.h"
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * A layer for transpose  a W x H x C to C x W x H, so that it can combine
 * with ResizeLayer to vectorize the image and perform perpixel operation
 * such as computing perpixel loss and normalize etc.
 *
 *
*/

class TransposeLayer : public Layer {
public:
  explicit TransposeLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void transpose(real* input,
                 real* output,
                 const std::vector<int> transOrder,
                 const bool is_forward);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

private:
  int batchSize_;
  int size_;
  int height_;
  int width_;
  int channel_;
  std::vector<int> transOrder_;
  bool needTrans_;
};

}  // namespace paddle
