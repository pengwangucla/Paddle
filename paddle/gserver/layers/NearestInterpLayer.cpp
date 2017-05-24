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
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

class NearestInterpLayer : public Layer {
protected:
  size_t outImgH_, outImgW_;
  size_t inImgH_, inImgW_;
  real ratioH_, ratioW_;
  size_t numChannels_;

public:
  explicit NearestInterpLayer(const LayerConfig& config) : Layer(config) {}

  virtual ~NearestInterpLayer() {}

  size_t getSize();
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override {}
};


REGISTER_LAYER(nearest_interp, NearestInterpLayer);

size_t NearestInterpLayer::getSize() {
  inImgH_ = inputLayers_[0]->getOutput().getFrameHeight();
  inImgW_ = inputLayers_[0]->getOutput().getFrameWidth();

  const NearestInterpConfig& conf = config_.inputs(0).nearest_interp_conf();
  if (inImgH_ == 0) {
    inImgH_ = conf.image_conf().img_size_y();
  }
  if (inImgW_ == 0) {
    inImgW_ = conf.image_conf().img_size();
  }

  outImgH_ = conf.out_size_y();
  outImgW_ = conf.out_size_x();
  numChannels_ = conf.image_conf().channels();

  CHECK(outImgH_ > 0 && outImgW_ > 0);
  CHECK(inImgH_ > 0 && inImgW_ > 0);
  CHECK(numChannels_);

  ratioH_ =
      (outImgH_ > 1) ? static_cast<real>(inImgH_) / (outImgH_) : 0.f;
  ratioW_ =
      (outImgW_ > 1) ? static_cast<real>(inImgW_) / (outImgW_) : 0.f;

  getOutput().setFrameHeight(outImgH_);
  getOutput().setFrameWidth(outImgW_);
  return outImgH_ * outImgW_ * numChannels_;
}

bool NearestInterpLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(1, config_.inputs_size());

  return true;
}

void NearestInterpLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t batchSize = getInput(0).getBatchSize();
  size_t size = getSize();
  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, size);
  }

  MatrixPtr inV = getInputValue(0);
  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwNearestInterpTimer", getName().c_str());
    if(useGpu_)
      outV->nearestForward(*inV,
                           inImgH_,
                           inImgW_,
                           outImgH_,
                           outImgW_,
                           numChannels_,
                           ratioH_,
                           ratioW_);
    else
      CHECK_EQ(1, 0)<<"Cpu of nearest is not implemented";
  }
}

}  // namespace paddle
