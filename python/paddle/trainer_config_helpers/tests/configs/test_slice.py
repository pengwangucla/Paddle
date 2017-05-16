from paddle.trainer_config_helpers import *

settings(batch_size=1)

data = data_layer(name='input', size=300, height=100, width=3)
slice = slice_layer(input=data, begin=0, size=50, axis=2)

outputs(slice)
