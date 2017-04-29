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

#include "Warp2DLayer.h"
#include "hl_warp2d.h"

namespace paddle {

REGISTER_LAYER(warp2d, Warp2DLayer);


bool Warp2DLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2UL);
  height_ = config_.height();
  width_ = config_.width();
  return true;
}


void Warp2DLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr image = getInputValue(0);
  MatrixPtr flow = getInputValue(1);
  batchSize_ = image->getHeight();
  size_ = image->getWidth();
  channel_ = size_ / height_ / width_;

  CHECK_EQ(flow->getWidth() / 2, height_ * width_);
  resizeOutput(batchSize_, size_);

  MatrixPtr warp_image = getOutputValue();

  if(useGpu_) {
    hl_warp2d_forward(image->getData(),
                      flow->getData(),
                      warp_image->getData(),
                      channel_,
                      height_,
                      width_);
  }
  else {
    CHECK_EQ(1, 0)<<"Not implemented";
  }
}

}  // namespace paddle
