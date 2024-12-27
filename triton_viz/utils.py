import inspect
import triton.language as tl
import numpy as np
import torch
import random
import sys
import math
import numpy as np
import json
import os
import triton
from functools import reduce
import sys
import inspect
import linecache
import pprint
import torch
import importlib
import sys
import operator
from .data import (
    Launch,
    Grid,
    Tensor,
    Load,
    Store,
    BinaryOp,
    MakeRange,
    ExpandDims,
    Dot,
    Reduce,
)

from triton.runtime.interpreter import (
    GridExecutor,
    _implicit_cvt,
    RESERVED_KWS,
    interpreter_builder,
    InterpretedFunction,
    TensorHandle,
    BlockPointerHandle
)




    
def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)


JS_MAX_SAFE_INTEGER = 9007199254740991  # 2^53 - 1

def get_tensor_ptr_map(base_ptr, tensor):
    """
    Handle 1D, 2D, or 3D tensors:
      1. Print the tensor shape.
      2. Compute a list (or nested list) of pointer addresses for all elements.
      3. Look up the first element's pointer in tensor_table and print debug info if found.
      4. Return the nested list of addresses.
    """


    # 1) Retrieve metadata
    shape       = tensor.shape        # e.g. (32,) or (32,48) or (2,3,4)
    stride      = tensor.stride()       # e.g. (48,1) for a 32x48
    elem_size   = tensor.element_size() # e.g. 4 (bytes) for fp32

    

    # 2) Build nested lists of element addresses
    addresses = {}

    # # ----- 1D TENSOR -----
    if len(shape) == 1:
        for i in range(shape[0]):
            offset = i * stride[0] * elem_size
            addresses[int(base_ptr + offset)] = [i,0,0]

    # # ----- 2D TENSOR -----
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                offset = (i * stride[0] + j * stride[1]) * elem_size
                addresses[int(base_ptr + offset)] = (i,j,0)

    # ----- 3D TENSOR -----
    elif len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    offset = (i * stride[0] + j * stride[1] + k * stride[2]) * elem_size
                    addresses[int(base_ptr + offset)] = [i,j,k]

    return addresses


def get_tensor_ptr(tensor,tensor_table):
    indices =[]
    base_ptr = None
    dims = len(tensor.handle.data.shape)
    full_tensor = None

    if dims == 1:
        base_ptr = tensor.handle.data[0]      
        for curr_tensor in tensor_table.values():  
            if base_ptr in curr_tensor.ptr_map:
                full_tensor = curr_tensor
                for x in range(0, len(tensor.handle.data)):

                    if tensor.handle.data[x] in set(curr_tensor.ptr_map.keys()):
                        indices.append(curr_tensor.ptr_map[tensor.handle.data[x]])
                    else:
                        print(tensor.shape)
                        print(tensor.handle.data)
                        print(f"Error Key {tensor.handle.data[x]}, (x:{x}) not found in {(full_tensor.var_name)}")
    if dims == 2:
        base_ptr = tensor.handle.data[0][0]        
        for curr_tensor in tensor_table.values():
            if base_ptr in curr_tensor.ptr_map:
                full_tensor = curr_tensor
                for x in range(0,len(tensor.handle.data)):
                    for y in range(0,len(tensor.handle.data[0])):
                        if int(tensor.handle.data[x][y]) in set(curr_tensor.ptr_map.keys()):
                            indices.append(curr_tensor.ptr_map[tensor.handle.data[x][y]])
                        else:
                            # print(int(tensor.handle.data[x][y]))
                            # print(tensor.shape)
                            # print(tensor.handle.data)
                            # print(curr_tensor.ptr_map)
                            # print( set(curr_tensor.ptr_map.keys()))
                            
                            print(f"Error Key {tensor.handle.data[x][y]}, (x:{x}, y:{y})  not found in {(full_tensor.var_name)}")

    if dims == 3:
        base_ptr = tensor.handle.data[0][0][0]     
        for curr_tensor in tensor_table.values():
            full_tensor = curr_tensor
            if base_ptr in curr_tensor.ptr_map:
                for x in range(0,len(tensor.handle.data)):
                    for y in range(0,len(tensor.handle.data[0])):
                        for z in range(0,len(tensor.handle.data[0][0])):
                            if tensor.handle.data[x][y][z] in set(curr_tensor.ptr_map.keys()):
                                indices.append(curr_tensor.ptr_map[tensor.handle.data[x][y][z]])
                            else:
                                print(f"Error Key {tensor.handle.data[x][y][z]}, (x:{x}, y:{y} z:{z}) not found in {(full_tensor.var_name)}")

    js_tensor = tensor_to_json(full_tensor)
                
    
    js_tensor['isTensorPtr']= True
    if indices != [(0, 0, 0)]:
        js_tensor["highlighted_indices"] = indices

    return js_tensor

    
    
def serialize(tensor,tensor_table):
    #return None
    if type(tensor) == triton.language.core.constexpr or type(tensor) == int:
        return int(tensor)

    elif type(tensor) == triton.language.core.tensor:
        if not hasattr(tensor, 'dtype'):
                printc(tensor,'red')


        dtype = tensor_to_json(tensor)['dtype']
        if "pointer" in dtype:
            
            return get_tensor_ptr(tensor,tensor_table)

        
        else:
            return get_tensor_data(tensor)    
    
    
    
def handle_large_numbers(value):
    """
    Convert large integers/floats beyond the JavaScript safe integer range
    into strings.
    """
    if isinstance(value, (int, float)):
        if abs(value) > JS_MAX_SAFE_INTEGER:
            return str(value)
    return value


def get_tensor_data(tensor):

    def constexpr_to_value(item):
        """Helper to recursively convert triton.language.core.constexpr to regular Python values."""
        if isinstance(item, triton.language.core.constexpr):
            return item.value
        elif isinstance(item, (list, tuple)):
            return [constexpr_to_value(i) for i in item]
        return item

    # If the tensor has a handle and exactly one element, treat as a scalar.
    if hasattr(tensor, 'handle') and len(tensor.handle.data) == 1:
        val = handle_special_floats(float(tensor.handle.data[0]))
        val = handle_large_numbers(val)
        return val
    
    js_tensor = {}
    if hasattr(tensor, 'numel'):
        js_tensor["numel"] = constexpr_to_value(tensor.numel)
    if hasattr(tensor, 'type'):
        js_tensor["type"] = str(tensor.type)
    if hasattr(tensor, 'shape'):
        js_tensor["shape"] = constexpr_to_value(tensor.shape)
    if hasattr(tensor, 'dtype'):
        js_tensor["dtype"] = str(tensor.dtype)

    js_tensor['isTensorPtr'] = False
    js_tensor['ptr_map'] = None

    if hasattr(tensor, 'handle'):
        js_tensor['value'] = numpy_to_json(tensor.handle.data)
    else:
        js_tensor['value'] = numpy_to_json(tensor.data)

    return js_tensor





def tensor_to_json(tensor):
    js_tensor ={}
    if hasattr(tensor, 'element_size'):
        js_tensor["element_size"] = int(tensor.element_size)
        
    if hasattr(tensor, 'dtype'):
        js_tensor["dtype"] = str(tensor.dtype)
    if hasattr(tensor, 'shape'):
        js_tensor["shape"] = list(tensor.shape)
    if hasattr(tensor, 'data'):
        js_tensor["data"] = numpy_to_json(tensor.data.numpy())
    return js_tensor


def numpy_to_json(arr):
    """
    Convert a Python/Numpy nested structure into a JSON-compatible format.
    Cast floats to float32 if applicable, and replace special floats / large numbers with strings.
    """
    if arr is None:
        return None

    # If arr is a numpy array
    if isinstance(arr, np.ndarray):
        # Convert to float32 if needed
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        # Handle different dimensions by recursing
        if arr.ndim == 0:
            val = handle_special_floats(arr.item())
            return handle_large_numbers(val)
        return [numpy_to_json(x) for x in arr]

    # If arr is a list, recursively process each element
    if isinstance(arr, list):
        return [numpy_to_json(x) for x in arr]

    # Handle numpy scalar types
    if isinstance(arr, (np.integer, np.bool_)):
        return arr.item()

    if isinstance(arr, (np.float64, np.float32)):
        val = handle_special_floats(float(arr))
        return handle_large_numbers(val)

    # Handle plain Python floats
    if isinstance(arr, float):
        val = handle_special_floats(arr)
        return handle_large_numbers(val)

    # Handle plain Python ints
    if isinstance(arr, int):
        return handle_large_numbers(arr)

    # All other types (str, etc.) just return as-is
    return arr




def handle_special_floats(value):
    """
    Handle special floating point values (Infinity, -Infinity, NaN)
    by converting them to JSON-compatible strings.
    """
    if isinstance(value, (float, np.float32, np.float64)):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        elif math.isnan(value):
            return "NaN"
    return value