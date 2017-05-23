*** Aborted at 1495239884 (unix time) try "date -d @1495239884" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGFPE (@0x7f0a530628fd) received by PID 20545 (TID 0x7f0a6d231740) from PID 1392912637; stack trace: ***
    @     0x7f0a6ce26330 (unknown)
    @     0x7f0a530628fd FLOAT_subtract
    @     0x7f0a5307abb0 PyUFunc_GenericFunction
    @     0x7f0a5307b4b6 ufunc_generic_call
    @           0x4c6d88 PyObject_CallFunctionObjArgs
    @           0x5381e8 (unknown)
    @           0x526a09 PyEval_EvalFrameEx
    @           0x5247ea PyEval_EvalFrameEx
    @           0x5247ea PyEval_EvalFrameEx
    @           0x567d14 (unknown)
    @           0x465bf4 PyRun_FileExFlags
    @           0x46612d PyRun_SimpleFileExFlags
    @           0x466d92 Py_Main
    @     0x7f0a6ca6ff45 __libc_start_main
    @           0x577c2e (unknown)
    @                0x0 (unknown)
Floating point exception (core dumped)


This is due to non-set value for gpu (FLOAT_infinite) is also due to outof float precision. Check your index of vector
wether you access the place where the gpu memory is not set.


