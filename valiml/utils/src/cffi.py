from os.path import dirname, join
from cffi import FFI
ffibuilder = FFI()

__dirname__ = dirname(__file__)

ffibuilder.set_source(
    "_utils",
    open(join(__dirname__, "utils.c")).read(),
    sources=[],
    libraries=["c"],
)

with open(join(__dirname__, "utils.h")) as f:
    ffibuilder.cdef(f.read())

if __name__ == "__main__":
    ffibuilder.compile()
