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
void doOneNearestInterpTest(MatrixPtr& tensor,
                        MatrixPtr& result,
                        size_t height,
                        size_t width,
                        size_t channel,
                        size_t out_height,
                        size_t out_width,
                        bool useGpu ){

  TestConfig config;
  config.layerConfig.set_type("nearest_interp");
  // config.layerConfig.set_height(height);
  // config.layerConfig.set_width(width);
  // config.layerConfig.set_size(height * width * channel);
  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", height * width * channel, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "nearest_interp",
      1, false, useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*tensor);

  NearestInterpConfig* nearest = input->mutable_nearest_interp_conf();
  ImageConfig* image = nearest->mutable_image_conf();
  image->set_img_size(width);
  image->set_img_size_y(height);
  image->set_channels(channel);
  nearest->set_out_size_x(out_width);
  nearest->set_out_size_y(out_height);

  std::vector<ParameterPtr> parameters;
  LayerPtr nearestInterpLayer;
  initTestLayer(config, &layerMap, &parameters, &nearestInterpLayer);
  nearestInterpLayer->forward(PASS_GC);
  MatrixPtr tmp = nearestInterpLayer->getOutputValue() ;
  tmp->print(std::cout);
  
  checkMatrixEqual(nearestInterpLayer->getOutputValue(), result);
}


TEST(Layer, NearestInterpLayer) {
  const int CHANNEL = 2;
  const int HEIGHT = 4;
  const int WIDTH = 2;
  const int OUT_HEIGHT = 4;
  const int OUT_WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;

  // TestConfig config;
  // config.biasSize = 0;
  // config.layerConfig.set_type("nearest_interp");
  // config.layerConfig.set_size(INPUT_SIZE);
  // config.layerConfig.set_height(HEIGHT);
  // config.layerConfig.set_width(WIDTH);
  // config.inputDefs.push_back({INPUT_DATA, "layer_0", INPUT_SIZE, 0});
  // LayerInputConfig* input = config.layerConfig.add_inputs();
  // NearestInterpConfig* nearest = input->mutable_nearest_interp_conf();

  MatrixPtr tensor, result;
  
  tensor = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real tensorData[] = {1, 2,
                       2, 2,
                       1, 2,
                       4, 4,
                       4, 2,
                       2, 4,
                       4, 4,
                       2, 2};
  tensor->setData(tensorData);

  result = Matrix::create(1, size_t(OUT_HEIGHT * OUT_WIDTH * CHANNEL),
                          false, false);
  real resultData[] = {1, 1, 2, 2,
                       2, 2, 2, 2,
                       1, 1, 2, 2,
                       4, 4, 4, 4,
                       4, 4, 2, 2,
                       2, 2, 4, 4,
                       4, 4, 4, 4,
                       2, 2, 2, 2};
  result->setData(resultData);

  //test forward correctness
  for (auto useGpu : {true}) {
    doOneNearestInterpTest(tensor, result,
      size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL),
      size_t(OUT_HEIGHT), size_t(OUT_WIDTH),
      useGpu);
    //use when test layer grad
    // nearest->set_out_size_x(OUT_WIDTH);
    // nearest->set_out_size_x(OUT_WIDTH);
  }
  LOG(INFO) << "Pass forward, test";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
