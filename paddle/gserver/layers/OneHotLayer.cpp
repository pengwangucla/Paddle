
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
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {

/*
 * Convert ids to one hot matrix
 * Might be useful when computing action for q network, combined with
 * random exploration (prob = onehot * (1-\alpha) + uniform * \alpha)
 */
class OneHotLayer : public Layer {
  public:
    explicit OneHotLayer(const LayerConfig& config) : Layer(config) {}

    void convert_real_to_int(const MatrixPtr& src, 
                             IVectorPtr& dst,
                             bool dst_use_gpu) {
        int height = src->getHeight();
        for(int i = 0 ; i < height; i ++) {
            real val = src->getElement(i, 0);
            int val_i = val - int(val) > 0.5 ? int(val) + 1 : int(val);
            dst->setElement(i, val_i);
        }
    }

    void forward(PassType passType) override {
        Layer::forward(passType);
        CHECK_EQ(inputLayers_.size(), 1UL);

        const MatrixPtr input = getInputValue(0);
        size_t class_num = getSize();
        int batchsize = input->getHeight();
        reserveOutput(batchsize, class_num);
        MatrixPtr output = getOutputValue();
        output->zeroMem();

        if(useGpu_ == false) {
            IVectorPtr ids = IVector::create(batchsize, false);
            convert_real_to_int(input, ids, false);
            auto cpu_sparse_val = ids->toOneHotSparseMatrix(
                class_num, false);
            output->copyFrom(*cpu_sparse_val);
        }
        else {
            MatrixPtr output = getOutputValue();
            hl_matrix_one_hot(input->getData(),
                              output->getData(),
                              batchsize,
                              class_num,
                              1.0);
        }
    }

    void backward(const UpdateCallback& callback) override {}
};





REGISTER_LAYER(one_hot, OneHotLayer);

} //namespace paddle
