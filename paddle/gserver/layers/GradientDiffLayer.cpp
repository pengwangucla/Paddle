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
 * @brief A layer for gradient diff layer
 * @note
 *    Input a layer h x w x c, and this layer compute gradient given different
 * steps
 *    It output a layer with shape h x w x (2*scale*c), and represents
 *    the gradient in x and y direction with the given scale.
 */

class GradientDiffLayer : public Layer {
public:
  explicit GradientDiffLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

private:
  int batchsize_;
  int size_;
  int height_;
  int width_;
  int channel_in_;
  int scale_num_;
  IVectorPtr scales;
};

REGISTER_LAYER(gradient_diff, GradientDiffLayer);
bool GradientDiffLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  // batchsize, channel, height, width
  CHECK_EQ(config_.has_width(), true);
  CHECK_EQ(config_.has_height(), true);
  height_ = config_.height();
  width_ = config_.width();

  auto& gradient_diff_conf = config_.inputs(0).gradient_diff_conf();
  scale_num_ = gradient_diff_conf.scales_size();
  scales = IVector::create(size_t(scale_num_), useGpu_);
  // copy scales
  for (int i = 0; i < scale_num_; i++)
    scales->setElement(i, gradient_diff_conf.scales(i));

  setNeedSequenceInfo(false);
  return true;
}

void GradientDiffLayer::forward(PassType passType) {
  Layer::forward(passType);
  const MatrixPtr input = getInputValue(0);
  batchsize_ = input->getHeight();
  size_ = input->getWidth();
  channel_in_ = size_ / height_ / width_;
  CHECK_EQ(0, size_ % (height_ * width_));
  reserveOutput(batchsize_, size_ * scale_num_ * 2);

  MatrixPtr output = getOutputValue();
  output->zeroMem();

  if (useGpu_) {
    hl_matrix_gradient_diff(input->getData(),
                            output->getData(),
                            scales->getData(),
                            scale_num_,
                            batchsize_ * size_,
                            height_,
                            width_,
                            channel_in_);
  } else {
    LOG(ERROR) << "Not implemented cpu gradient diff layer";
  }
}

void GradientDiffLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  const Argument& input = getInput(0);
  MatrixPtr outputGrad = getOutputGrad();

  if (outputGrad == NULL || input.grad == NULL) {
    return;
  }

  // the grad should be rotated in the reverse direction
  MatrixPtr preGrad = getInputGrad(0);
  MatrixPtr tmpGrad = Matrix::create(batchsize_, size_, false, useGpu_);
  tmpGrad->zeroMem();

  if (useGpu_) {
    hl_matrix_gradient_diff_derivative(outputGrad->getData(),
                                       tmpGrad->getData(),
                                       input.value->getData(),
                                       scales->getData(),
                                       scale_num_,
                                       batchsize_ * size_,
                                       height_,
                                       width_,
                                       channel_in_);
    preGrad->add(*tmpGrad);
  } else {
    LOG(ERROR) << "Not implemented cpu gradient diff";
  }
}

}  // namespace paddle
