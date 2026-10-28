"""Keep CPU below normal, without lowering this process's memory/I/O priority."""
import ctypes as C
from ctypes import wintypes as W


def enter_background_mode():
    k=C.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=W.HANDLE
    k.SetPriorityClass.argtypes=[W.HANDLE,W.DWORD];k.SetPriorityClass.restype=W.BOOL
    k.GetPriorityClass.argtypes=[W.HANDLE];k.GetPriorityClass.restype=W.DWORD
    p=k.GetCurrentProcess();ok=k.SetPriorityClass(p,0x4000)
    if not ok:raise C.WinError(C.get_last_error())
    import psutil
    current=psutil.Process()
    current.ionice(psutil.IOPRIO_NORMAL)
    actual_io=int(current.ionice())
    if actual_io!=int(psutil.IOPRIO_NORMAL):raise RuntimeError('Normal training I/O priority was not applied')
    return {'background_processing_mode':False,'cpu_below_normal':True,
        'memory_priority':'inherited','io_priority':'normal explicitly verified','io_priority_value':actual_io,
        'effective_priority_class':int(k.GetPriorityClass(p))}
