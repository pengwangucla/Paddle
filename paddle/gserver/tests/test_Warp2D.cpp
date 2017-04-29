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
void doOneWarpTest(const MatrixPtr& tensor,
                   const MatrixPtr& flow,
                   const MatrixPtr& result,
                   size_t height,
                   size_t width,
                   bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("warp2d");
  config.layerConfig.set_height(height);
  config.layerConfig.set_width(width);

  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", height * width, 0});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back(
    {INPUT_DATA, "layer_1", 2 * height * width, 0});
  config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "warp2d",
      1, false, useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*tensor);
  dataLayers[1]->getOutputValue()->zeroMem();
  dataLayers[1]->getOutputValue()->copyFrom(*flow);

  std::vector<ParameterPtr> parameters;
  LayerPtr warp2dLayer;
  initTestLayer(config, &layerMap, &parameters, &warp2dLayer);
  warp2dLayer->forward(PASS_GC);
  
  checkMatrixEqual(warp2dLayer->getOutputValue(), result);
}

TEST(Layer, Warp2DLayer) {
  bool useGpu = true;
  FLAGS_use_gpu = useGpu;

  const int CHANNEL = 1;
  const int HEIGHT = 4;
  const int WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;

  MatrixPtr tensor, flow, result;
  tensor = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real tensorData[] = {1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2};

  tensor->setData(tensorData);

  flow = Matrix::create(1, size_t(2 * HEIGHT * WIDTH),
                        false, false);
  real flowData[] = {-1, -1, -1, -1, 
                     -1, -1, -1, -1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0};
  flow->setData(flowData);

  result = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real resultData[] = {0, 1, 2, 2,
                       0, 1, 2, 4,
                       2, 2, 4, 0,
                       4, 2, 2, 0};

  result->setData(resultData);
  doOneWarpTest(tensor, flow, result, size_t(HEIGHT), size_t(WIDTH), useGpu);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
