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

#include "TransposeLayer.h"

namespace paddle {

REGISTER_LAYER(transpose, TransposeLayer);

bool compareVector(const std::vector<int>& left,
                   const std::vector<int>& right) {
  if (left.size() != right.size()) return false;
  for (int i = 0; i < int(left.size()); i++) {
    if (left[i] != right[i]) return false;
  }

  return true;
}

void TransposeLayer::transpose(real* input,
                               real* output,
                               const std::vector<int> transOrder,
                               const bool is_forward) {
  // first using the existing matrix operation for Gpu speed
  int heightNew = channel_, widthNew = height_ * width_;
  if (compareVector(transOrder, std::vector<int>{1, 2, 0})) {
    heightNew = is_forward ? channel_ : height_ * width_;
    widthNew = is_forward ? height_ * width_ : channel_;
  } else if (compareVector(transOrder, std::vector<int>{2, 0, 1})) {
    heightNew = is_forward ? height_ * channel_ : width_;
    widthNew = is_forward ? width_ : height_ * channel_;
  } else {
    LOG(ERROR) << "Not implemented";
    CHECK_EQ(1, 0);
  }

  MatrixPtr inputSample =
      Matrix::create(input, heightNew, widthNew, false, useGpu_);
  MatrixPtr outputSample =
      Matrix::create(output, widthNew, heightNew, false, useGpu_);
  if (is_forward) {
    inputSample->transpose(outputSample, false);
  } else {
    MatrixPtr tmpGrad = nullptr;
    inputSample->transpose(tmpGrad, true);
    outputSample->add(*tmpGrad);
  }
}

bool TransposeLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  // original order is C x H x W ()
  // new order is first second and third order, same with np.transpose
  CHECK_EQ(inputLayers_.size(), 1UL);
  transOrder_.push_back(config_.inputs(0).transpose_conf().trans_order_c());
  transOrder_.push_back(config_.inputs(0).transpose_conf().trans_order_h());
  transOrder_.push_back(config_.inputs(0).transpose_conf().trans_order_w());

  height_ = config_.height();
  width_ = config_.width();

  for (int i = 0; i < 3; i++) {
    if (transOrder_[i] != i) {
      needTrans_ = true;
      break;
    }
  }

  return true;
}

void TransposeLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr input = getInputValue(0);
  batchSize_ = input->getHeight();
  size_ = input->getWidth();
  CHECK_GE(size_, height_ * width_);
  CHECK_EQ(size_ % (height_ * width_), 0)
      << "total size_ is not dividable by (height_ * width_), i.e., "
      << "channel number should be an integer";
  channel_ = size_ / (height_ * width_);

  resizeOutput(batchSize_, size_);

  // (TODO:peng) later modify to more generalized transpose ()
  MatrixPtr outV = getOutputValue();
  if (!needTrans_) {
    outV->assign(*input);
    return;
  }
  for (int b = 0; b < batchSize_; b++) {  // for each input feat map
    transpose(input->getData() + b * size_,
              outV->getData() + b * size_,
              transOrder_,
              true);
  }

  if (getInputGrad(0)) {
    zeroGrad();
  }
}

void TransposeLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = getOutputGrad();
  if (outputGrad == NULL) {
    return;
  }
  // the grad should be rotated in the reverse direction
  MatrixPtr preGrad = getInputGrad(0);

  if (!needTrans_) {
    preGrad->add(*outputGrad);
    return;
  }

  for (int b = 0; b < batchSize_; b++) {  // for each input feat map
    transpose(outputGrad->getData() + b * size_,
              preGrad->getData() + b * size_,
              transOrder_,
              false);
  }
}

}  // namespace paddle
