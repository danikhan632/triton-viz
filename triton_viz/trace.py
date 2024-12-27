from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction
import traceback
from .interpreter import patch
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


        self.src =kernel.src
        self._fn = InterpretedFunction(kernel.fn)

    def run(self, *args, **kwargs):
        with patch():
            self._fn.run(*args, **kwargs)


    def get_src(self):
        print(self.src)
        return self.src

def trace(kernel):
    return Trace(kernel)

