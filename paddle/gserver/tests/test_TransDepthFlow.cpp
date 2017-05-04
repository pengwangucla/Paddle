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
void doDepth2FlowTest(const MatrixPtr& input,
                   const MatrixPtr& trans,
                   const MatrixPtr& output,
                   size_t height,
                   size_t width,
                   bool useGpu,
                   bool depth2flow) {
  TestConfig config;
  config.layerConfig.set_type("trans_depth_flow");
  config.layerConfig.set_height(height);
  config.layerConfig.set_width(width);


  size_t size = depth2flow ? height * width : 2 * height * width ;

  config.inputDefs.push_back({INPUT_DATA, "layer_0", size, 0});
  LayerInputConfig* input_config = config.layerConfig.add_inputs();

  TransDepthFlowConfig* trans_depth_flow = 
            input_config->mutable_trans_depth_flow_conf();
  trans_depth_flow->set_depth_to_flow(depth2flow);

  config.inputDefs.push_back({INPUT_DATA, "layer_1", size_t(10), 0});
  config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "trans_depth_flow",
      1, false, useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*input);
  dataLayers[1]->getOutputValue()->zeroMem();
  dataLayers[1]->getOutputValue()->copyFrom(*trans);

  std::vector<ParameterPtr> parameters;
  LayerPtr transDepthFlowLayer;
  initTestLayer(config, &layerMap, &parameters, &transDepthFlowLayer);
  transDepthFlowLayer->forward(PASS_GC);
  
  checkMatrixEqual(transDepthFlowLayer->getOutputValue(), output);
}


TEST(Layer, Warp2DLayer) {
  bool useGpu = true;
  FLAGS_use_gpu = useGpu;

  const int HEIGHT = 4;
  const int WIDTH = 4;
  const int INPUT_SIZE = HEIGHT * WIDTH;

  MatrixPtr depth, flow, trans;
  depth = Matrix::create(1, size_t(INPUT_SIZE), false, false);
  real depthData[] = {1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2};
  // this should be the inverse of translation of camera
  depth->setData(depthData);

  // the first relative to the second camera
  trans = Matrix::create(1, size_t(10), false, false);
  real transData[] = {1, 1, 2, 2, 0, 0, 0, 1, 1, 0};
  trans->setData(transData);

  flow = Matrix::create(1, size_t(2 * INPUT_SIZE), false, false);
  real flowData[] = { 1.  ,  0.5 ,  0.5 ,  0.5,
        1.  ,  0.5 ,  0.25,  0.25,
        0.25,  0.5 ,  0.5 ,  0.25,
        0.25,  0.25,  0.5 ,  0.5 ,
        1.  ,  0.5 ,  0.5 ,  0.5 ,
        1.  ,  0.5 ,  0.25,  0.25,
        0.25,  0.5 ,  0.5 ,  0.25,
        0.25,  0.25,  0.5 ,  0.5};
  flow->setData(flowData);

  doDepth2FlowTest(depth, trans,
   flow, size_t(HEIGHT), size_t(WIDTH), useGpu, true);
  LOG(INFO)<<"TEST depth to flow finished";

  doDepth2FlowTest(flow, trans,
   depth, size_t(HEIGHT), size_t(WIDTH), useGpu, false);
  LOG(INFO)<<"TEST flow to depth finished";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
