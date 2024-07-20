from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
model_dir = "out_sleepedf/train/4/best_ckpt"
checkpoint_path = os.path.join(model_dir, "best_model.ckpt-71682")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
for key in var_to_shape_map:#list the keys of the model
    # print(key)
    # print(reader.get_tensor(key))
    shape = np.shape(reader.get_tensor(key))  #get the shape of the tensor in the model
    shape = list(shape)
    # print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim
    # print(variable_parameters)
    total_parameters += variable_parameters

print(total_parameters)