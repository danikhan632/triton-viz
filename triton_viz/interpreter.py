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
from .utils import (
    get_tensor_ptr_map,
    serialize
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
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime import JITFunction
from typing import Tuple, List, Optional
from contextlib import contextmanager
from functools import wraps


record_builder=[]
tensor_table = {}


data = []
def get_data():
    return data


def _patch_lang(fn):
    triton_patch_lang(fn)






def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)

def _unpatch_lang():


    if tl.__name__ in sys.modules:
        importlib.reload(tl)



def _check_storage_contiguous(tensor):
    # Note that this is different from if a tensor is accessed contiguously, so we cannot use tensor.is_contiguous()
    # 1. Sort strides from smallest to largest
    # 2. If the tensor is contiguous, the stride product should be the same of the shape product of all previous dimensions
    shape_prod = 1
    indices = sorted(range(len(tensor.stride())), key=tensor.stride().__getitem__)
    for i, index in enumerate(indices):
        stride = tensor.stride(index)
        shape = tensor.shape[index]
        if i == 0 and stride != 1:
            return False
        if i != 0 and stride != shape_prod:
            return False
        shape_prod *= shape
    return True


def format_tensor(args,constexprs):
    tensors = []
    call_args = {}
    for name, arg in args.items():
        # printc((name, arg))
        
        if name in constexprs:
            call_args[name] = arg
        else:
            ret = _implicit_cvt(arg)
            if hasattr(arg, "data_ptr"):
                assert _check_storage_contiguous(
                    arg
                ), "triton-viz only supports contiguouly stored tensors for now"
                base_ptr = int(ret.handle.data[0])
                tmp = Tensor(base_ptr,ret.dtype, arg.stride(), arg.shape, arg.element_size(), arg, name, get_tensor_ptr_map(base_ptr,arg), None)
                
                tensors.append(
                    tmp
                )
                tensor_table[base_ptr] = tmp
                
            call_args[name] = ret
            
            
    return call_args, tensors










def _grid_executor_call(self, *args_dev, **kwargs):
    
    # Store the code object of this function for identification
    code_obj = _grid_executor_call.__code__

    # Dictionary to keep track of previous local variable values
    prev_locals = {}
    UNDEFINED = object()

    # Variable to hold the current block indices
    current_block_indices = [0, 0, 0]

  
    filename = [None]
    last_line = [None]  # Just store the last line info

    def trace_func(frame, event, arg):  

        if event == 'line':

            if filename[0] is None:
                filename[0] = frame.f_code.co_filename

            if filename[0] in frame.f_code.co_filename:
                # Store current line info for potential last line
                last_line[0] = {
                    'lineno': frame.f_lineno,
                    'locals': frame.f_locals.copy()
                }

                lineno = frame.f_lineno
                source_line = linecache.getline(filename[0], lineno).strip()

                local_vars = frame.f_locals.copy()
                
                # printc(source_line)
                # Compute changed variables
                changed_vars = {}
                for var_name, value in local_vars.items():

                    try:
                        prev_safe = None
                        prev_value = prev_locals.get(var_name, UNDEFINED)
                        if prev_value is UNDEFINED:
                            changed_vars[var_name] = None
                        else:
      
                            prev_safe = serialize(prev_value,tensor_table)
                            
                        new_safe = serialize(prev_value,tensor_table)
                        

                        if prev_safe != None or prev_safe != new_safe:
                            changed_vars[var_name] = new_safe
                            
                    except Exception as e:
                        print(f"Error: Could not convert variable '{var_name}'. Exception: {str(e)}")


                data.append({
                    'source_line': source_line,
                    'changed_vars': changed_vars,
                    'block_indices': current_block_indices,
                    'line_number': lineno
                })

                # Update prev_locals
                prev_locals.clear()
                prev_locals.update(local_vars)
                # print(data)
                
        return trace_func

    try:
        
        # Removes reserved keywords from kwargs
        argspec = inspect.getfullargspec(self.fn)
        kwargs = {k: v for k, v in kwargs.items() if k in argspec.args}
        if kwargs.pop("warmup", False):
            return
        args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
        # Remaps core language functions to interpreted ones
        _patch_lang(self.fn)
        # Prepare call arguments
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
        
        call_args, tensors = format_tensor(args,self.constexprs)
        # print(tensors)
    
        call_args.pop("self", None)
        # Iterate through grid
        grid = self.grid(call_args) if callable(self.grid) else self.grid
        printc(call_args)
        assert len(grid) <= 3
        grid = grid + (1,) * (3 - len(grid))
        interpreter_builder.set_grid_dim(*grid)
  

        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    interpreter_builder.set_grid_idx(x, y, z)
                    record_builder.append([x, y, z])
                    current_block_indices = [x, y, z]

                    try:
                        sys.settrace(trace_func)
                        self.fn(**call_args)
                        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
                    finally:
                        sys.settrace(None)


        # Copy arguments back to propagate side-effects
        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
        _unpatch_lang()
    finally:
        return data


def _jit_function_call(self, *args, **kwargs):
    triton_patch_lang(self.fn)
    return self.fn(*args, **kwargs)



@contextmanager
def patch():
    old_grid_executor_call = GridExecutor.__call__
    #printc(old_grid_executor_call, 'green')
    old_jit_function_call = JITFunction.__call__
    # XXX(Keren): Temporarily disable rewriting of AST
    # old_rewrite_ast = InterpretedFunction._rewrite_ast

    GridExecutor.__call__ = _grid_executor_call
    JITFunction.__call__ = _jit_function_call
    InterpretedFunction._rewrite_ast = lambda self: self.fn
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call


