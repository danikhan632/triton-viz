import inspect
import triton.language as tl
import numpy as np
import torch
import random
import sys
import math
import numpy as np
import json
import triton
from functools import reduce
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
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime import JITFunction
from typing import Tuple, List, Optional
from contextlib import contextmanager
from functools import wraps

JS_MAX_SAFE_INTEGER = 9007199254740991  # 2^53 - 1


data = []
def get_data():
    return data


def _patch_lang(fn):
    triton_patch_lang(fn)
    tl.sum = _create_reduce(tl.reduce, "sum")
    tl.min = _create_reduce(tl.reduce, "min")
    tl.max = _create_reduce(tl.reduce, "max")


def _unpatch_lang():
    import importlib
    import sys

    if tl.__name__ in sys.modules:
        importlib.reload(tl)

class RecordBuilder:
    
    def to_dict(self):
        return {
            "launches": [
                {
                    "grid": launch.grid,
                    "tensors": [tensor.__dict__ for tensor in launch.tensors],
                    "records": [
                        {
                            **record.__dict__,
                            "lhs_name": getattr(record, "lhs_name", None),
                            "rhs_name": getattr(record, "rhs_name", None)
                        } if isinstance(record, BinaryOp) else record.__dict__
                        for record in launch.records
                    ]
                }
                for launch in self._launches
            ]
        }


    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self._launches: List[Launch] = []
        self._sampling_grid_idx: Optional[Tuple] = None
        self._grid_idx = (0, 0, 0)
        self._grid_dim = (1, 1, 1)

    @property
    def launches(self):
        return self._launches

    def set_sampling_grid_idx(self, idx: Tuple):
        self._sampling_grid_idx = idx

    def set_grid_dim(self, nx, ny, nz):
        self._grid_dim = (nx, ny, nz)
        self._launches.append(Launch((nx, ny, nz), [], []))

    def set_grid_idx(self, x, y, z):
        assert x < self._grid_dim[0]
        assert y < self._grid_dim[1]
        assert z < self._grid_dim[2]
        self._grid_idx = (x, y, z)
        grid_record = Grid(self._grid_idx)
        self.add_record(grid_record)

    def add_tensor(self, data, dtype, shape=None, stride=None):
        tensor = Tensor(data, shape, stride, dtype)
        self._launches[-1].tensors.append(tensor)

    def add_tensors(self, tensors):
        self._launches[-1].tensors.extend(tensors)

    def sort_tensor_handles(self):
        # Sort tensor handles based on ptr
        launch = self._launches[-1]
        launch.tensors = sorted(launch.tensors, key=lambda x: x.ptr)

    def get_tensor_ptr(self, ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self._launches[-1].tensors)):
            if ptr < self._launches[-1].tensors[i].ptr:
                break
            ret_idx = i
        return self._launches[-1].tensors[ret_idx]

    def add_record(self, record):
        def _to_1d_grid(idx: Tuple):
            # Assuming originally 1d, 2d, or 3d input
            if len(idx) == 1:
                return idx[0]
            elif len(idx) == 2:
                return idx[0] * self._grid_dim[1] + idx[1]
            elif len(idx) == 3:
                return (
                    idx[0] * self._grid_dim[1] * self._grid_dim[2]
                    + idx[1] * self._grid_dim[2]
                    + idx[2]
                )

        if not self._sampling_grid_idx or _to_1d_grid(
            self._sampling_grid_idx
        ) == _to_1d_grid(self._grid_idx):
            self._launches[-1].records.append(record)


record_builder = RecordBuilder()


def collect_grid():
    for launch in record_builder.launches[-1:]:
        records, tensor_table, failures = collect_launch(launch)
    return records, tensor_table, failures

def collect_launch(launch):
    tensor_table = {}
    for i, t in enumerate(launch.tensors):
        tensor_table[t.ptr] = (t, i)
    failures = {}
    all_grids = {}
    last_grid = None
    program_records = []
    for r in launch.records:
        if isinstance(r, Grid):
            if last_grid is not None:
                all_grids[last_grid.idx] = program_records
                program_records = []
            last_grid = r
        program_records.append(r)
        if (
            isinstance(r, (Store, Load))
            and (r.invalid_access_masks & r.original_masks).any()
        ):
            failures[last_grid.idx] = True
    all_grids[last_grid.idx] = program_records
    return all_grids, tensor_table, failures

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

def convert_numpy_types(obj):
    import math
    if isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(element) for element in obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.generic):
        value = obj.item()
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None  # Replace with None or 'Infinity', 'NaN' as strings if preferred
            else:
                return value
        else:
            return value
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # Replace with None or 'Infinity', 'NaN' as strings if preferred
        else:
            return obj
    else:
        return obj
    
    
def get_dims(data):
    if np.isscalar(data) or (isinstance(data, list) and len(data) == 1):
        return [-1, -1, -1]
    elif isinstance(data, (list, np.ndarray)):
        shape = np.array(data).shape
        if len(shape) == 1:
            return [shape[0], -1, -1]
        elif len(shape) == 2:
            return [shape[0], shape[1], -1]
        elif len(shape) >= 3:
            return list(shape[:3])
    return [-1, -1, -1]  # Default case for unrecognized data types




def check_tensor_element(data, tensor, dim):
    data = int(data)
    ptr = int(tensor.ptr)
    
    # Calculate total size for any number of dimensions
    total_elements = reduce(operator.mul, tensor.shape[:dim], 1)
    total_size = total_elements * tensor.element_size
    
    start = ptr
    end = ptr + total_size
    
    is_within_range = start <= data < end
    
    return is_within_range









def compute_coords(ptr, base_ptr, element_size, strides, shape, dims):
    """
    Computes the coordinates in the source tensor given a pointer.
    
    Args:
        ptr (int): The pointer value.
        base_ptr (int): The base pointer of the source tensor.
        element_size (int): The size of each element in bytes.
        strides (list): The strides of the tensor.
        shape (list): The shape of the tensor.
        dims (int): The number of dimensions of the tensor.

    Returns:
        tuple: A tuple (x, y, z) representing the coordinates in the tensor.
    """
    # Calculate the offset in bytes from the base pointer
    byte_offset = ptr - base_ptr

    # Convert byte offset to element offset
    element_offset = byte_offset // element_size

    # Calculate the coordinates
    x = element_offset // strides[0]
    remainder = element_offset % strides[0]
    y = remainder // strides[1]
    z = remainder % strides[1]

    # Fill unused dimensions with -1
    coords = [x, y, z]
    for k in range(3 - dims):
        coords[-(k + 1)] = -1  # Fill trailing dimensions with -1
    return tuple([int(coords[0]),int(coords[1]),int(coords[2])])


def process_array(array, base_ptr, element_size, strides, shape, dims):
    """
    Processes a NumPy array of pointers and computes the corresponding coordinates in the source tensor.

    Args:
        array (numpy.ndarray): A NumPy array of pointers, which can be 1D, 2D, or 3D.
        base_ptr (int): The base pointer of the source tensor.
        element_size (int): The size of each element in bytes.
        strides (list): The strides of the tensor.
        shape (list): The shape of the tensor.
        dims (int): The number of dimensions of the tensor.

    Returns:
        list: A nested list matching the shape of the input array, where each element is
              a tuple (x, y, z) representing the coordinates.
    """

    if array.ndim == 0:  # Scalar pointer (0D array)
        return compute_coords(array.item(), base_ptr, element_size, strides, shape, dims)
    elif array.ndim == 1:  # 1D array of pointers
        return [compute_coords(array[i], base_ptr, element_size, strides, shape, dims)
                for i in range(array.shape[0])]
    elif array.ndim == 2:  # 2D array of pointers
        return [[compute_coords(array[i, j], base_ptr, element_size, strides, shape, dims)
                 for j in range(array.shape[1])]
                for i in range(array.shape[0])]
    elif array.ndim == 3:  # 3D array of pointers
        return [[[compute_coords(array[i, j, k], base_ptr, element_size, strides, shape, dims)
                  for k in range(array.shape[2])]
                 for j in range(array.shape[1])]
                for i in range(array.shape[0])]
    else:
        raise ValueError("ptrs_array must be a NumPy array with 1D, 2D, or 3D shape.")


def map_pointers_to_coords(source_tensor, ptrs_array):
    """
    Maps a pointer array to coordinates in the source tensor using the tensor's base pointer,
    strides, and element size.

    Args:
        source_tensor (torch.Tensor): The source tensor (1D, 2D, or 3D).
        ptrs_array (numpy.ndarray): A NumPy array where each element is a pointer.

    Returns:
        list: A nested list (matching ptrs_array shape) of tuples (x, y, z) representing the coordinates
              in the source tensor. Unused dimensions are filled with -1.
    """
    # Extract tensor metadata
    base_ptr = source_tensor.ptr  # Base pointer address
    element_size = source_tensor.element_size  # Size of each element in bytes
    strides = list(source_tensor.stride)  # Strides for each dimension
    shape = list(source_tensor.shape)  # Shape of the source tensor
    dims = len(shape)  # Number of dimensions (1, 2, or 3)

    # Extend strides and shape to always have 3 dimensions
    strides.extend([1] * (3 - dims))
    shape.extend([1] * (3 - dims))

    # Process the array
    res = process_array(ptrs_array, base_ptr, element_size, strides, shape, dims)

    return res



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

def handle_large_numbers(value):
    """
    Convert large integers/floats beyond the JavaScript safe integer range
    into strings.
    """
    if isinstance(value, (int, float)):
        if abs(value) > JS_MAX_SAFE_INTEGER:
            return str(value)
    return value

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


def to_json(tensor):
    def constexpr_to_value(item):
        if isinstance(item, triton.language.core.constexpr):
            return item.value
        elif isinstance(item, (list, tuple)):
            return [constexpr_to_value(i) for i in item]
        return item

    return {
        "shape": constexpr_to_value(tensor.shape),
        "numel": constexpr_to_value(tensor.numel),
        "dtype": str(tensor.dtype),
        "type": str(tensor.type),
    }


def serialize(tensor):

    if type(tensor) == triton.language.core.constexpr:
        return int(tensor)
    elif type(tensor) == int:
        return tensor
    elif type(tensor) == triton.language.core.tensor:
        dtype = to_json(tensor)['dtype']
        shape = to_json(tensor)['shape']

        data = {}

        # Handle pointer-based tensors
        if "pointer" in dtype:
            if len(shape) == 0:
                rec_tensor = record_builder.get_tensor_ptr(tensor.handle.data[0])
                data['tensor_ptr'] = str(rec_tensor.ptr)
                data['dim'] = rec_tensor.data.shape
                data['solo_ptr'] = True
            
            if len(shape) == 1:
                rec_tensor = record_builder.get_tensor_ptr(tensor.handle.data[0])
                tensor_coords = map_pointers_to_coords(rec_tensor, tensor.handle.data)
                data['tensor_ptr'] = str(rec_tensor.ptr)
                data['dim'] = rec_tensor.data.shape
                data['slice_shape'] = shape
                data['highlighted_indices'] = tensor_coords
                data['solo_ptr'] = False

            if len(shape) == 2:
                rec_tensor = record_builder.get_tensor_ptr(tensor.handle.data[0][0])
                tensor_coords = map_pointers_to_coords(rec_tensor, tensor.handle.data)
                data['tensor_ptr'] = str(rec_tensor.ptr)
                data['shape'] = rec_tensor.data.shape
                data['slice_shape'] = shape
                data['highlighted_indices'] = tensor_coords
                # Convert the numeric data using numpy_to_json (which handles special and large values)
                data['value'] = numpy_to_json(rec_tensor.data.tolist())
                data['solo_ptr'] = False

            if len(shape) == 3:
                rec_tensor = record_builder.get_tensor_ptr(tensor.handle.data[0][0][0])
                tensor_coords = map_pointers_to_coords(rec_tensor, tensor.handle.data)
                data['tensor_ptr'] = str(rec_tensor.ptr)
                data['shape'] = rec_tensor.data.shape
                data['slice_shape'] = shape
                data['highlighted_indices'] = tensor_coords
                data['value'] = numpy_to_json(rec_tensor.data.tolist())
                data['solo_ptr'] = False

            data['isTensorPtr']=True

            return data
        else:
            # Non-pointer-based tensor
            if len(tensor.handle.data) == 1:
                # Scalar value
                val = handle_special_floats(float(tensor.handle.data[0]))
                val = handle_large_numbers(val)
                return val
            data = to_json(tensor)
            data['isTensorPtr']=False

            data['value'] = numpy_to_json(tensor.handle.data)
            return data
    else:
        # If it's some other type, just return as is or convert if needed
        return tensor


    
def normalize_tensor(c):
    # Ensure c is a tensor
    c = torch.tensor(c, dtype=torch.float32)

    # Calculate min and max of the tensor
    min_c = torch.min(c)
    max_c = torch.max(c)
    # Normalize each element in the tensor
    normalized_c = c / (max_c - min_c)
    
    return normalized_c




def get_ir():
    """
    Retrieves the cached IR from the compile process.
    Assumes the cache has been initialized and populated during compilation.
    """
    import os
    import json
    from pathlib import Path

    return open("/workspaces/triton-viz/examples/matmul.ttgir","r").read()


def _grid_executor_call(self, *args_dev, **kwargs):
    import sys
    import inspect
    import linecache
    import pprint
    import torch

    # Store the code object of this function for identification
    code_obj = _grid_executor_call.__code__

    # Dictionary to keep track of previous local variable values
    prev_locals = {}
    UNDEFINED = object()

    # Variable to hold the current block indices
    current_block_indices = (0, 0, 0)

    data = []
    filename = [None]
    last_line = [None]  # Just store the last line info

    def trace_func(frame, event, arg):
        # We only want to record 'line' events from the user code
        # i.e. when the interpreter executes a new line of Python code
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

                # Compute changed variables
                changed_vars = {}
                for var_name, value in local_vars.items():
                    try:
                        prev_value = prev_locals.get(var_name, UNDEFINED)
                        if prev_value is UNDEFINED:
                            prev_safe = None
                        else:
                            prev_safe = serialize(prev_value)
                        new_safe = serialize(value)

                        if prev_safe != new_safe:
                            changed_vars[var_name] = new_safe
                    except Exception as e:
                        print(f"Error: Could not convert variable '{var_name}'. Exception: {str(e)}")

                # Capture IR or any other useful data here
                # Assume we can retrieve IR by calling get_ir()
                # This will store the current IR after each line event
                ir = get_ir()  # assume this function returns IR in some form

                # Additional useful data could be included, for example:
                current_filename = frame.f_code.co_filename
                current_function = frame.f_code.co_name

                data.append({
                    'source_line': source_line,
                    'changed_vars': changed_vars,
                    'block_indices': current_block_indices,
                    'ir': ir,
                    'filename': current_filename,
                    'function_name': current_function,
                    'line_number': lineno
                })

                # Update prev_locals
                prev_locals.clear()
                prev_locals.update(local_vars)

        return trace_func

    try:
        # --- Original _grid_executor_call code starts here ---

        # Removes reserved keywords from kwargs
        kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
        src_map = kwargs.pop("src_map")
        src = kwargs.pop("src")

        if kwargs.pop("warmup", False):
            return
        args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
        # Remaps core language functions to interpreted ones
        _patch_lang(self.fn)
        # Prepare call arguments
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
        call_args = {}
        tensors = []
        for name, arg in args.items():
            if name in self.constexprs:
                call_args[name] = arg
            else:
                try:
                    # Attempt to convert arg to tensor
                    ret = _implicit_cvt(arg)
                    if hasattr(arg, "data_ptr"):
                        assert _check_storage_contiguous(
                            arg
                        ), f"Only supports contiguously stored tensors, but '{name}' is not contiguous."
                        tensors.append(
                            Tensor(
                                ret.handle.data[0],
                                ret.dtype,
                                arg.stride(),
                                arg.shape,
                                arg.element_size(),
                                arg
                            )
                        )
                except AssertionError as e:
                    raise
                except Exception as e:
                    raise

                call_args[name] = ret

        call_args.pop("self", None)
        # Iterate through grid
        grid = self.grid(call_args) if callable(self.grid) else self.grid
        assert len(grid) <= 3
        grid = grid + (1,) * (3 - len(grid))
        interpreter_builder.set_grid_dim(*grid)
        record_builder.set_grid_dim(*grid)
        record_builder.add_tensors(tensors)
        record_builder.sort_tensor_handles()

        current_block_indices = (0, 0, 0)

        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    current_block_indices = (x, y, z)
                    interpreter_builder.set_grid_idx(x, y, z)
                    record_builder.set_grid_idx(x, y, z)
                    try:
                        sys.settrace(trace_func)
                        self.fn(**call_args)
                        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
                    finally:
                        sys.settrace(None)
                        # After each block execution, if we have last line info, add it
                        if last_line[0] is not None:
                            source_line = linecache.getline(filename[0], last_line[0]['lineno']).strip()
                            changed_vars = {}
                            for var_name, value in last_line[0]['locals'].items():
                                try:
                                    new_safe = serialize(value)
                                    if new_safe is not None:
                                        changed_vars[var_name] = new_safe
                                except Exception:
                                    continue

                            # Capture IR or any other data here again if needed at block end
                            dire = get_ir()
                            printc(dire,"magenta")

                            data.append({
                                'source_line': source_line,
                                'changed_vars': changed_vars,
                                'block_indices': current_block_indices,
                                'is_final': True,
                                'ir': ir
                            })
        _unpatch_lang()

    finally:
        return data




def _get_variable_name(var):
    import inspect
    import re
    
    # Get the calling frame
    frame = inspect.currentframe().f_back.f_back
    
    # Get the source code of the calling frame
    source = inspect.getsource(frame)
    
    # Find all assignments in the source code
    assignments = re.findall(r'(\w+)\s*=\s*' + re.escape(str(var)), source)
    
    if assignments:
        return assignments[0]
    else:
        # If we can't find an exact match, look for partial matches
        partial_matches = re.findall(r'(\w+)\s*=.*' + re.escape(str(var)), source)
        return partial_matches[0] if partial_matches else str(var)

def _jit_function_call(self, *args, **kwargs):
    triton_patch_lang(self.fn)
    return self.fn(*args, **kwargs)


def check_out_of_bounds_access(ptrs, masks):
    first_ptr = np.reshape(ptrs.data, (-1))[0]
    tensor_ptr = record_builder.get_tensor_ptr(first_ptr)
    offsets = ptrs.data - tensor_ptr.ptr
    max_valid_offset = np.prod(tensor_ptr.shape) * tensor_ptr.element_size
    valid_access_masks = (offsets >= 0) & (offsets < max_valid_offset)
    invalid_access_masks = (~valid_access_masks) & masks.data
    corrected_offsets = np.where(valid_access_masks, offsets, 0)
    return (
        tensor_ptr,
        valid_access_masks & masks.data,
        invalid_access_masks,
        corrected_offsets,
        offsets,
    )


def _create_masked_load(fn):
    @wraps(fn)
    def wrapper(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        load_record = Load(
            ptr=tensor_ptr.ptr,
            shape=ptrs.data.shape,
            offsets=corrected_offsets,
            access_masks=valid_access_masks,
            invalid_access_masks=invalid_access_masks,
            original_offsets=original_offsets,
            original_masks=masks.data,
        )
        record_builder.add_record(load_record)

        return fn(
            ptrs,
            masks,
            other,
            cache_modifier,
            eviction_policy,
            is_volatile,
        )

    return wrapper


def _create_masked_store(fn):
    @wraps(fn)
    def wrapper(ptrs, value, masks, cache_modifier, eviction_policy):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        store_record = Store(
            ptr=tensor_ptr.ptr,
            shape=ptrs.data.shape,
            offsets=corrected_offsets,
            access_masks=valid_access_masks,
            invalid_access_masks=invalid_access_masks,
            original_offsets=original_offsets,
            original_masks=masks.data,
        )
        record_builder.add_record(store_record)

        return fn(ptrs, value, valid_access_masks, cache_modifier, eviction_policy)

    return wrapper


def _create_make_range(fn):
    @wraps(fn)
    def wrapper(start, stop):
        range_record = MakeRange(start=start, end=stop)
        record_builder.add_record(range_record)
        return fn(start, stop)

    return wrapper

def _create_binary_op(fn):
    @wraps(fn)
    def wrapper(lhs, rhs, op):
        ret = fn(lhs, rhs, op)
        
        binary_op_record = BinaryOp(
            op=op.__name__,
            input_shape=(lhs.data.shape),
            output_shape=ret.data.shape,
            lhs_name=lhs,
            rhs_name=rhs,
            operator=op
        )
        record_builder.add_record(binary_op_record)
        return ret

    return wrapper




def _create_dot(fn):
    @wraps(fn)
    def wrapper(a, b, c, allow_tf32, max_num_imprecise_acc):
        dot_record = Dot(
            input_shape=a.data.shape,
            other_shape=b.data.shape,
            output_shape=c.data.shape,
            input_data=a.data.tolist(),
            other_data=b.data.tolist()
        )
        record_builder.add_record(dot_record)

        def capture_intermediate(row, col, result):
            dot_record.update_intermediate(row, col, float(result))

        # Modify the original function to call capture_intermediate at each step
        def modified_fn(a, b, c, allow_tf32, max_num_imprecise_acc):
            for i in range(a.data.shape[0]):
                for j in range(b.data.shape[1]):
                    A_row = a.data[i, :]
                    B_column = b.data[:, j]
                    result = np.dot(A_row, B_column)
                    capture_intermediate(i, j, result)
                    c.data[i, j] = result

        modified_fn(a, b, c, allow_tf32, max_num_imprecise_acc)
        return c

    return wrapper

def _create_expand_dims(fn):
    @wraps(fn)
    def wrapper(arg, axis):
        ret = fn(arg, axis)
        expand_dims_record = ExpandDims(
            input_shape=arg.data.shape, index=axis, output_shape=ret.data.shape
        )
        record_builder.add_record(expand_dims_record)
        return ret

    return wrapper


def _create_reduce(fn, op_name: str):
    @wraps(fn)
    def wrapper(input, axis=None, keep_dims=False):
        mapping = {
            "max": tl.standard._elementwise_max,
            "min": tl.standard._elementwise_min,
            "sum": tl.standard._sum_combine,
        }
        ret = fn(input, axis=axis, combine_fn=mapping[op_name], keep_dims=keep_dims)
        reduce_record = Reduce(
            input_shape=input.handle.data.shape,
            index=axis,
            op=op_name,
            keep_dims=keep_dims,
            output_shape=ret.handle.data.shape,
        )
        record_builder.add_record(reduce_record)
        return ret

    return wrapper


@contextmanager
def patch():
    old_grid_executor_call = GridExecutor.__call__
    #printc(old_grid_executor_call, 'green')
    old_jit_function_call = JITFunction.__call__
    # XXX(Keren): Temporarily disable rewriting of AST
    # old_rewrite_ast = InterpretedFunction._rewrite_ast

    old_create_make_range = interpreter_builder.create_make_range
    old_create_masked_load = interpreter_builder.create_masked_load
    old_create_expand_dims = interpreter_builder.create_expand_dims
    old_binary_op = interpreter_builder.binary_op
    old_create_dot = interpreter_builder.create_dot
    old_create_masked_store = interpreter_builder.create_masked_store
    GridExecutor.__call__ = _grid_executor_call
    JITFunction.__call__ = _jit_function_call
    InterpretedFunction._rewrite_ast = lambda self: self.fn
    interpreter_builder.create_make_range = _create_make_range(
        interpreter_builder.create_make_range
    )
    interpreter_builder.create_masked_load = _create_masked_load(
        interpreter_builder.create_masked_load
    )
    interpreter_builder.create_expand_dims = _create_expand_dims(
        interpreter_builder.create_expand_dims
    )
    interpreter_builder.binary_op = _create_binary_op(interpreter_builder.binary_op)
    interpreter_builder.create_dot = _create_dot(interpreter_builder.create_dot)
    interpreter_builder.create_masked_store = _create_masked_store(
        interpreter_builder.create_masked_store
    )
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call
        # InterpretedFunction._rewrite_ast = old_rewrite_ast
        interpreter_builder.create_make_range = old_create_make_range
        interpreter_builder.create_masked_load = old_create_masked_load
        interpreter_builder.create_expand_dims = old_create_expand_dims
        interpreter_builder.binary_op = old_binary_op
        interpreter_builder.create_dot = old_create_dot
        interpreter_builder.create_masked_store = old_create_masked_store



def get_recorded_data():
    """Return the recorded data in a format suitable for JSON serialization."""
    return record_builder.to_dict()