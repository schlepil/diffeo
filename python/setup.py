from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Build import cythonize  

from Cython.Compiler import Options
Options. fast_fail = True 

setup(ext_modules = cythonize([Extension("diffeoPy",
                                       ["diffeoPy.pyx"], 
                                       include_dirs = ["../cpp", "/usr/include/eigen3"],
                                       library_dirs=["/usr/lib/x86_64-linux-gnu"],
                                       extra_compile_args=['-O3', '-std=c++11'],
                                       extra_link_args=['-ltinyxml'],
                                       language="c++")])
)