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
void doOneGradientDiffTest(MatrixPtr& tensor,
                           MatrixPtr& result,
                           std::vector<int> scales,
                           size_t batchsize,
                           size_t height,
                           size_t width,
                           size_t channel,
                           bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("gradient_diff");
  config.layerConfig.set_height(height);
  config.layerConfig.set_width(width);
  config.layerConfig.set_size(height * width * channel);

  config.inputDefs.push_back(
      {INPUT_DATA, "layer_0", height * width * channel, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(config,
                &dataLayers,
                &datas,
                &layerMap,
                "gradient_diff",
                batchsize,
                false,
                useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*tensor);

  GradientDiffConfig* gradient_diff = input->mutable_gradient_diff_conf();
  for (size_t i = 0; i < scales.size(); i++)
    gradient_diff->add_scales(scales[i]);

  std::vector<ParameterPtr> parameters;
  LayerPtr gradientDiffLayer;
  initTestLayer(config, &layerMap, &parameters, &gradientDiffLayer);
  gradientDiffLayer->forward(PASS_GC);

  // gradientDiffLayer->getOutputValue()->print(std::cout);
  // result->print(std::cout);
  checkMatrixEqual(gradientDiffLayer->getOutputValue(), result);
}

TEST(Layer, GradientDiffLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("gradient_diff");
  const int CHANNEL = 1;
  const int HEIGHT = 4;
  const int WIDTH = 4;
  const int BATCH_SIZE = 2;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;

  config.layerConfig.set_size(INPUT_SIZE);
  config.layerConfig.set_height(HEIGHT);
  config.layerConfig.set_width(WIDTH);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", INPUT_SIZE, 0});

  LayerInputConfig* input = config.layerConfig.add_inputs();
  GradientDiffConfig* gradient_diff = input->mutable_gradient_diff_conf();

  MatrixPtr tensor;
  MatrixPtr result_gradient_diff_scale1;

  tensor =
      Matrix::create(1, size_t(BATCH_SIZE) * size_t(INPUT_SIZE), false, false);
  real tensorData[] = {1, 2, 2, 2, 1, 2, 4, 4, 4, 2, 2, 4, 4, 4, 2, 2,

                       4, 2, 2, 4, 4, 4, 2, 2, 1, 2, 2, 2, 1, 2, 4, 4};
  tensor->setData(tensorData);

  result_gradient_diff_scale1 = Matrix::create(
      size_t(BATCH_SIZE), size_t(INPUT_SIZE) * size_t(2), false, false);

  real GradientDiffBatchData[] = {
      1. / 3.,  0,       0,       0.,       1. / 3.,  2. / 6.,  0.,       0.,
      -2. / 6., 0.,      2. / 6., 0.,       0.,       -2. / 6,  0.,       0.,

      0.,       0.,      2. / 6., 2. / 6.,  3. / 5.,  0.,       -2. / 6., 0.,
      0.,       2. / 6,  0.,      -2. / 6,  0.,       0.,       0.,       0.,

      -2 / 6.,  0.,      2. / 6., 0.,       0.,       -2. / 6,  0.,       0.,
      1. / 3,   0.,      0.,      0.,       1. / 3.,  2. / 6.,  0.,       0.,

      0.,       2. / 6., 0.,      -2. / 6., -3. / 5., -2. / 6., 0.,       0.,
      0.,       0.,      2. / 6., 2. / 6.,  0.,       0.,       0.,       0.};
  result_gradient_diff_scale1->setData(GradientDiffBatchData);

  // test forward correctness
  for (auto useGpu : {true}) {
    doOneGradientDiffTest(tensor,
                          result_gradient_diff_scale1,
                          std::vector<int>(1, 1),
                          size_t(BATCH_SIZE),
                          size_t(HEIGHT),
                          size_t(WIDTH),
                          size_t(CHANNEL),
                          useGpu);

    LOG(INFO) << "Test gradient diff scale 1, useGpu " << useGpu;

    gradient_diff->add_scales(1);
    gradient_diff->add_scales(2);
    testLayerGrad(config, "gradient_diff", 1, false, useGpu);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
