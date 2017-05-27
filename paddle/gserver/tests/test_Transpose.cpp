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
#include "paddle/gserver/layers/TransposeLayer.h"
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
void doOneTransposeTest(MatrixPtr& tensor,
                        std::vector<int> transOrder,
                        MatrixPtr& result,
                        size_t height,
                        size_t width,
                        size_t channel,
                        bool useGpu ){

  TestConfig config;
  config.layerConfig.set_type("transpose");

  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", height * width * channel, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "transpose",
      1, false, useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*tensor);

  TransposeConfig* transpose = input->mutable_transpose_conf();
  ImageConfig* image = transpose->mutable_image_conf();
  image->set_channels(channel);
  image->set_img_size(width);
  image->set_img_size_y(height);

  transpose->set_trans_order_c(transOrder[0]);
  transpose->set_trans_order_h(transOrder[1]);
  transpose->set_trans_order_w(transOrder[2]);

  std::vector<ParameterPtr> parameters;
  LayerPtr transposeLayer;
  initTestLayer(config, &layerMap, &parameters, &transposeLayer);
  transposeLayer->forward(PASS_GC);
  MatrixPtr tmp = transposeLayer->getOutputValue();
  tmp->print(std::cout);

  checkMatrixEqual(transposeLayer->getOutputValue(), result);
}


TEST(Layer, TransposeLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("transpose");
  const int CHANNEL = 3;
  const int HEIGHT = 4;
  const int WIDTH = 2;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;
  std::vector<int> order0 = std::vector<int>{1, 2, 0},
                   order1 = std::vector<int>{2, 0, 1};

  config.layerConfig.set_size(INPUT_SIZE);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", INPUT_SIZE, 0});

  LayerInputConfig* input = config.layerConfig.add_inputs();
  TransposeConfig* transpose = input->mutable_transpose_conf();

  MatrixPtr tensor, result120, result201;
  
  tensor = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real tensorData[] = {1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       4, 2, 2, 4,
                       4, 4, 2, 2};
  tensor->setData(tensorData);

  result120 = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real result120Data[] = {1, 4, 4,
                          2, 2, 2,
                          2, 2, 2,
                          2, 4, 4,
                          1, 4, 4,
                          2, 4, 4,
                          4, 2, 2,
                          4, 2, 2};
  result120->setData(result120Data);

  //test forward correctness
  for (auto useGpu : {false, true}) {
    doOneTransposeTest(tensor, order0, result120,
      size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL), useGpu);
    doOneTransposeTest(result120, order1, tensor,
      size_t(WIDTH), size_t(CHANNEL), size_t(HEIGHT), useGpu);
  }
  LOG(INFO) << "Pass forward, test";

  //test backward correctness
  for (auto useGpu : {false, true}) {
    ImageConfig* image = transpose->mutable_image_conf();
    image->set_img_size(WIDTH);
    image->set_img_size_y(HEIGHT);
    transpose->set_trans_order_c(order0[0]);
    transpose->set_trans_order_h(order0[1]);
    transpose->set_trans_order_w(order0[2]);
    testLayerGrad(config, "transpose", 10, false, useGpu);

    transpose->set_trans_order_c(order1[0]);
    transpose->set_trans_order_h(order1[1]);
    transpose->set_trans_order_w(order1[2]);
    testLayerGrad(config, "transpose", 10, false, useGpu);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
