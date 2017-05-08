/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/gserver/layers/Warp2DLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_double(checkgrad_eps);
DECLARE_bool(thread_local_rand_use_global_seed);
DECLARE_bool(prev_batch_state);

// Test that the convTrans forward is the same as conv backward
void doOneWarpTest(const MatrixPtr& tesnor,
                   const MatrixPrt& flow,
                   const MatrixPrt& result,
                   const int height,
                   const int width,
                   const bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type('warp2d');
  config.layerConfig.set_height(height);
  config.layerConfig.set_width(width);

  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", height * width, 0})
  config.LayerConfig.add_inputs();
  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", 2 * height * width, 0})
  config.LayerConfig.add_inputs();

  LayerMap layerMap;
  std::vector<ParameterPtr> parameters;

  LayerPtr warp2dLayer;

  (*warp2dLayer) = Layer::create(config.layerConfig);
  (*layerMap)[config.layerConfig.name()] = *warp2dLayer;
  (*warp2dLayer)->init((*layerMap), parameterMap);
  (*warp2dLayer)->setNeedGradient(false);

  MatrixPtr input = warp2dLayer->getInputValue(0);
  input = Matrix::create(1, height * width, false, useGpu);
  input->copyFrom(tensor);
  MatrixPtr input_2 = warp2dLayer->getInputValue(1);
  input2 = Matrix::create(1, 2 * height * width, false, useGpu);
  input2->copyFrom(flow);

  warp2dLayer->forward(PASS_TEST);
  checkMatrixEqual(warp2dLayer->getOutputValue(), result);
}

TEST(Layer, Warp2DLayer) {
  MatrixPtr tensor, flow, result;
  bool useGpu = true;
  bool FLAGS_use_gpu = useGpu;
  const int CHANNEL = 1;
  const int HEIGHT = 4;
  const int WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;

  tensor = Matrix::create(1, INPUT_SIZE, false, useGpu);
  real tensorData[] = {1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2}

  tensor->setData(resultData);

  flow = Matrix::create(1, 2 * HEIGHT * WIDTH, false, useGpu);
  real flowData[] = {-1, -1, -1, -1, 
                     -1, -1, -1, -1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0}
  flow->setData(flowData);

  result = Matrix::create(1, INPUT_SIZE, false, useGpu);
  real resultData[] = {0, 1, 2, 2,
                       0, 1, 4, 4,
                       2, 2, 4, 0,
                       4, 2, 2, 0}

  result->setData(resultData);

  doOneWarpTest(tensor, flow, result, HEIGHT, WIDTH);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
