from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction
import traceback
from .interpreter import patch, record_builder
from typing import Tuple

# Global variable to store dat
_global_blocks = []
_global_src =""

def get_src():
    global _global_src
    return _global_src



def get_blocks():
    global _global_blocks
    return _global_blocks

class Trace(KernelInterface):
    def __init__(self, kernel: JITFunction) -> None:
        self.src = kernel.src
        self.src_map = {}
        assert isinstance(kernel, JITFunction), "Kernel must be a JITFunction"
        self._fn = InterpretedFunction(kernel.fn)

    def run(self, *args, **kwargs):
        global _global_blocks
        global _global_src
        with patch():
            kwargs['src_map'] = self.src_map
            kwargs['src'] = self.src
            _global_src = self.src
            dat = self._fn.run(*args, **kwargs)
            _global_blocks = dat  
            return dat

    def get_src(self):
        return self.src

def trace(kernel):
    return Trace(kernel)
