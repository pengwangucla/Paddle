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

#include "TransDepthFlowLayer.h"
#include "hl_warp2d.h"

#include "paddle/utils/Logging.h"
namespace paddle {

REGISTER_LAYER(trans_depth_flow, TransDepthFlowLayer);

bool TransDepthFlowLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for trans-depth-flow is 2 */
  CHECK_EQ(config_.inputs_size(), 2UL);
  height_ = config_.height();
  width_ = config_.width();

  return true;
}

void TransDepthFlowLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr input = getInputValue(0);
  MatrixPtr trans = getInputValue(1);

  CHECK_EQ(trans->getWidth(), 10)<<"trans must be a 10 dim vector";
  
  batchSize_ = input->getHeight();
  size_ = input->getWidth();
  channel_ = size_ / height_ / width_;

  bool depth_to_flow = 
     config_.inputs(0).trans_depth_flow_conf().depth_to_flow();

  if(depth_to_flow) {
    CHECK_EQ(channel_, 1);
    resizeOutput(batchSize_, size_ * 2);
  }
  else {
    CHECK_EQ(channel_, 2);
    resizeOutput(batchSize_, size_ / 2);
  }
  MatrixPtr output = getOutputValue();

  // input->print(std::cout);

  if(useGpu_) {
    hl_trans_depth_flow_forward(input->getData(),
                      trans->getData(),
                      output->getData(),
                      batchSize_,
                      height_,
                      width_,
                      depth_to_flow);
    // LOG(INFO)<<"result";
    // output->print(std::cout);
  }
  else {
    CHECK_EQ(1, 0)<<"CPU version is not implemented";
  }
}

}  // namespace paddle
