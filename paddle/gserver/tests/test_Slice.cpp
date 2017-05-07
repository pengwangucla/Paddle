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
void doOneSliceTest(MatrixPtr& tensor,
                    MatrixPtr& result,
                    int begin_idx,
                    int slice_size,
                    int slice_axis,
                    size_t batchsize,
                    size_t height,
                    size_t width,
                    size_t channel,
                    bool useGpu){

  TestConfig config;
  config.layerConfig.set_type("slice");
  config.layerConfig.set_height(height);
  config.layerConfig.set_width(width);
  config.layerConfig.set_size(height * width * channel);

  config.inputDefs.push_back(
    {INPUT_DATA, "layer_0", height * width * channel, 0});
  LayerInputConfig* input = config.layerConfig.add_inputs();

  std::vector<DataLayerPtr> dataLayers;
  LayerMap layerMap;
  vector<Argument> datas;
  initDataLayer(
      config, &dataLayers, &datas, &layerMap, "slice",
      batchsize, false, useGpu);

  dataLayers[0]->getOutputValue()->zeroMem();
  dataLayers[0]->getOutputValue()->copyFrom(*tensor);

  SliceConfig* slice = input->mutable_slice_conf();
  slice->set_begin(begin_idx);
  slice->set_size(slice_size);
  slice->set_axis(slice_axis);

  std::vector<ParameterPtr> parameters;
  LayerPtr sliceLayer;
  initTestLayer(config, &layerMap, &parameters, &sliceLayer);
  sliceLayer->forward(PASS_GC);
  
  checkMatrixEqual(sliceLayer->getOutputValue(), result);

}


TEST(Layer, SliceLayer) {
  TestConfig config;
  config.biasSize = 0;
  config.layerConfig.set_type("slice");
  const int CHANNEL = 2;
  const int HEIGHT = 4;
  const int WIDTH = 4;
  const int BATCH_SIZE = 2;
  const int INPUT_SIZE = HEIGHT * WIDTH * CHANNEL;

  config.layerConfig.set_size(INPUT_SIZE);
  config.layerConfig.set_height(HEIGHT);
  config.layerConfig.set_width(WIDTH);
  config.inputDefs.push_back({INPUT_DATA, "layer_0",
    INPUT_SIZE, 0});

  LayerInputConfig* input = config.layerConfig.add_inputs();
  SliceConfig* slice = input->mutable_slice_conf();

  MatrixPtr tensor;
  MatrixPtr result_slice_batch;
  MatrixPtr result_slice_channel;
  MatrixPtr result_slice_height;
  MatrixPtr result_slice_width;
  
  tensor = Matrix::create(1, size_t(BATCH_SIZE) *size_t(INPUT_SIZE),
   false, false);
  real tensorData[] = {1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       1, 2, 2, 2,
                       1, 2, 4, 4,

                       1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       1, 2, 2, 2,
                       1, 2, 4, 4};
  tensor->setData(tensorData);

  int axis = 0, begin=0, size=1;
  result_slice_batch = Matrix::create(size, 
    size_t(INPUT_SIZE), false, false);

  real sliceBatchData[] = {1, 2, 2, 2,
                           1, 2, 4, 4,
                           4, 2, 2, 4,
                           4, 4, 2, 2,
                           4, 2, 2, 4,
                           4, 4, 2, 2,
                           1, 2, 2, 2,
                           1, 2, 4, 4};
  result_slice_batch->setData(sliceBatchData);

  //test forward correctness
  for (auto useGpu : {true}) {
    doOneSliceTest(tensor, result_slice_batch,
                   begin, size, axis, size_t(BATCH_SIZE),
                   size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL), useGpu);
    LOG(INFO)<<"Test slice batch, useGpu "<<useGpu;
    slice->set_axis(axis);
    slice->set_begin(begin);
    slice->set_size(size);
    testLayerGrad(config, "slice", 10, false, useGpu);
  }


  axis = 1;
  begin = 0;
  size = 1;
  result_slice_channel = Matrix::create(BATCH_SIZE, 
    size_t(size * INPUT_SIZE / CHANNEL), false, false);

  real sliceChannelData[] = {1, 2, 2, 2,
                             1, 2, 4, 4,
                             4, 2, 2, 4,
                             4, 4, 2, 2,
                             1, 2, 2, 2,
                             1, 2, 4, 4,
                             4, 2, 2, 4,
                             4, 4, 2, 2};

  result_slice_channel->setData(sliceChannelData);

  doOneSliceTest(tensor, result_slice_channel,
                 begin, size, axis, size_t(BATCH_SIZE),
                 size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL), true);
  LOG(INFO)<<"Test slice channel";

  slice->set_axis(axis);
  slice->set_begin(begin);
  slice->set_size(size);
  testLayerGrad(config, "slice", 10, false, true);


  axis = 2;
  begin = 1;
  size = 2;
  result_slice_height = Matrix::create(BATCH_SIZE, 
    size_t(size * INPUT_SIZE / HEIGHT), false, false);

  real sliceHeightData[] = {1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       1, 2, 2, 2,
                       1, 2, 4, 4,
                       4, 2, 2, 4,
                       4, 4, 2, 2,
                       1, 2, 2, 2};
  result_slice_height ->setData(sliceHeightData);
  
  doOneSliceTest(tensor, result_slice_height,
                 begin, size, axis, size_t(BATCH_SIZE),
                 size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL), true);
  LOG(INFO)<<"Test slice height";

  slice->set_axis(axis);
  slice->set_begin(begin);
  slice->set_size(size);
  testLayerGrad(config, "slice", 10, false, true);


  axis = 3;
  begin = 1;
  size = 2;
  result_slice_width = Matrix::create(BATCH_SIZE,
    size_t(size * INPUT_SIZE / WIDTH), false, false);

  real sliceWidthData[] = { 2, 2,
                            2, 4,
                            2, 2,
                            4, 2,
                            2, 2,
                            4, 2,
                            2, 2,
                            2, 4,
                            2, 2,
                            2, 4,
                            2, 2,
                            4, 2,
                            2, 2,
                            4, 2,
                            2, 2,
                            2, 4};
  result_slice_width->setData(sliceWidthData);
  
  doOneSliceTest(tensor, result_slice_width,
                 begin, size, axis, size_t(BATCH_SIZE),
                 size_t(HEIGHT), size_t(WIDTH), size_t(CHANNEL), true);
  LOG(INFO)<<"Test slice width";

  slice->set_axis(axis);
  slice->set_begin(begin);
  slice->set_size(size);
  testLayerGrad(config, "slice", 10, false, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
