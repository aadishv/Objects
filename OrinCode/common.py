#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    # Get number of bindings using the tensor-centric API in TensorRT 8+
    num_bindings = engine.num_io_tensors
    
    for i in range(num_bindings):
        # Get tensor name and properties
        tensor_name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(tensor_name)
        shape = engine.get_tensor_shape(tensor_name)
        
        # Calculate size (including batch dimension)
        size = trt.volume(shape) * dtype.itemsize
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        
        # Check if tensor is input or output using tensor-centric API
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # This function is kept for backward compatibility, but it's recommended to use do_inference_v2
    return do_inference_v2(context, bindings, inputs, outputs, stream)

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # Run inference - properly set tensor addresses for TensorRT 8+
    if hasattr(context, 'execute_async_v3'):
        # For TensorRT 8+ we need to explicitly set tensor addresses
        for i, inp in enumerate(inputs):
            tensor_name = context.engine.get_tensor_name(i)
            context.set_tensor_address(tensor_name, int(inp.device))
            
        for i, out in enumerate(outputs):
            tensor_name = context.engine.get_tensor_name(i + len(inputs))
            context.set_tensor_address(tensor_name, int(out.device))
            
        context.execute_async_v3(stream_handle=stream.handle)
    elif hasattr(context, 'execute_async_v2'):
        # TensorRT 7.x
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    else:
        # Older versions
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs
    return [out.host for out in outputs]
