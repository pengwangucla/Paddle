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
#include "paddle/math/Matrix.h"

namespace paddle {

class GradientStoppingLayer : public Layer {
public:
  GradientStoppingLayer(const LayerConfig& config) : Layer(config) {}

  virtual bool init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
    bool ret = Layer::init(layerMap, parameterMap);
    // CHECK_EQ(1UL, inputLayers_.size());
    return ret;
  }

  virtual void forward(PassType passType) {
    Layer::forward(passType);
    // get the last input and propogate it forward as output
    const Argument& input = getInput(inputLayers_.size() - 1);
    size_t batchSize = input.getBatchSize();
    Matrix::resizeOrCreate(
        output_.value, batchSize, input.value->getWidth(), false, useGpu_);

    // forward: identity function
    output_.value->assign(*input.value);
  }

  virtual void backward(const UpdateCallback& callback) {
    // this is intended to be an empty method
    return;
  }
};

REGISTER_LAYER(gradient_stopping, GradientStoppingLayer);

}  // namespace paddle
