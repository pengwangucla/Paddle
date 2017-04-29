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
TEST(Layer, TransposeLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("transpose");
  const int CHANNEL = 2;
  const int HEIGHT = 8;
  const int WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;
  config.layerConfig.set_size(INPUT_SIZE);
  config.layerConfig.set_height(HEIGHT);
  config.layerConfig.set_width(WIDTH);
  config.inputDefs.push_back({INPUT_DATA, "layer_0", INPUT_SIZE, 0});

  for (auto useGpu : {false, true}) {
    testLayerGrad(config, "transpose", 100, false, useGpu);
  }

}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
