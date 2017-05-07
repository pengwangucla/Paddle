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

#include "Layer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {
/**
 * @brief A layer for resizing a minibatch matrix
 * @note
 * origin matrix height * width)
 * resize matrix: (height * width / size) * size
 */

class SliceLayer : public Layer {
public:
  explicit SliceLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

private:
  int batchsize_;
  int size_;
  int height_;
  int width_;
  IVectorPtr shape;

  int begin_idx_;
  int slice_size_;
  int slice_axis_;
  int out_size_;
};


REGISTER_LAYER(slice, SliceLayer);
bool SliceLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  // batchsize, channel, height, width
  shape = IVector::create(size_t(4), useGpu_);
  shape->zeroMem();

  height_ = config_.height();
  width_ = config_.width();
  shape->setElement(2, config_.height());
  shape->setElement(3, config_.width());
  CHECK_GT(shape->getElement(2), 0);
  CHECK_GT(shape->getElement(3), 0);

  begin_idx_ = config_.inputs(0).slice_conf().begin();
  slice_size_ = config_.inputs(0).slice_conf().size();
  slice_axis_ = config_.inputs(0).slice_conf().axis();

  setNeedSequenceInfo(false);
  return true;
}

void SliceLayer::forward(PassType passType) {
  Layer::forward(passType);
  const MatrixPtr input = getInputValue(0);
  batchsize_ = input->getHeight();
  size_ = input->getWidth();
  shape->setElement(0, input->getHeight());
  shape->setElement(1, size_ / height_ /width_);

  out_size_ = 1;
  for(int i = 0; i < 4; i ++)
    out_size_ *= (i == slice_axis_? slice_size_ :shape->getElement(i));

  CHECK_GE(shape->getElement(slice_axis_),
    slice_size_ + begin_idx_);

  if(slice_axis_ == 0) {
    reserveOutput(slice_size_, size_);
    MatrixPtr output = getOutputValue();

    output->copyFrom(input->getData() + begin_idx_ * size_,
             size_t(slice_size_ * size_));
  } else {
    reserveOutput(batchsize_, 
      slice_size_ * size_ / shape->getElement(slice_axis_));

    MatrixPtr output = getOutputValue();

    if(useGpu_){
      hl_matrix_slice(input->getData(),
        output->getData(),
        shape->getData(),
        out_size_,
        begin_idx_,
        slice_size_,
        slice_axis_,
        true);
    }
    else {
      LOG(ERROR)<< "Not implemented cpu slice for axis > 0";
    }
  }
}

void SliceLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  const Argument& input = getInput(0);
  MatrixPtr outputGrad = getOutputGrad();

  if (outputGrad == NULL || input.grad == NULL) {
    return;
  }

  // the grad should be rotated in the reverse direction
  MatrixPtr preGrad = getInputGrad(0);

  MatrixPtr tmpGrad = Matrix::create(batchsize_, size_,
     false, useGpu_);
  tmpGrad->zeroMem();

  if(slice_axis_ == 0) {
    if(useGpu_) {
      hl_matrix_slice(outputGrad->getData(),
        tmpGrad->getData(),
        shape->getData(),
        out_size_,
        begin_idx_,
        slice_size_,
        slice_axis_,
        false);
    } else {
      real* src = outputGrad->getData();
      real* dst = tmpGrad->getData() + begin_idx_ * size_;
      memcpy(dst, src, sizeof(real) * slice_size_ * size_);
    }
    preGrad->add(*tmpGrad);
  } else {
    if(useGpu_) {
      hl_matrix_slice(outputGrad->getData(),
        tmpGrad->getData(),
        shape->getData(),
        out_size_,
        begin_idx_,
        slice_size_,
        slice_axis_,
        false);
      preGrad->add(*tmpGrad);
    }
    else {
      LOG(ERROR)<< "Not implemented cpu slice for axis > 0";
    }
  }
}

}  // namespace paddle
