"""Assign a suspended Windows launcher to the job before any descendants exist."""
import ctypes as C
from ctypes import wintypes as W
import os
import subprocess
import sys
import msvcrt
import _winapi


class SuspendedChild:
    def __init__(self,argv,*,output,cwd,environment):
        read_fd,write_fd=os.pipe()
        out_handle=msvcrt.get_osfhandle(output.fileno())
        os.set_inheritable(read_fd,True);os.set_handle_inheritable(out_handle,True)
        si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESTDHANDLES
        si.hStdInput=msvcrt.get_osfhandle(read_fd);si.hStdOutput=out_handle;si.hStdError=out_handle
        try:
            self._handle,self._thread,self.pid,self.tid=_winapi.CreateProcess(
                argv[0],subprocess.list2cmdline(argv),None,None,True,
                subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS|4,
                environment,cwd,si)
        except BaseException:
            os.close(write_fd);raise
        finally:
            os.close(read_fd);os.set_handle_inheritable(out_handle,False)
        self.stdin=os.fdopen(write_fd,'wb');self.returncode=None

    def resume(self):
        kernel=C.WinDLL('kernel32',use_last_error=True)
        kernel.ResumeThread.argtypes=[W.HANDLE];kernel.ResumeThread.restype=W.DWORD
        if kernel.ResumeThread(self._thread)!=1:raise C.WinError(C.get_last_error())
        _winapi.CloseHandle(self._thread);self._thread=None

    def poll(self):
        if _winapi.WaitForSingleObject(self._handle,0)==0:self.returncode=_winapi.GetExitCodeProcess(self._handle)
        return self.returncode

    def wait(self,timeout=30):
        if _winapi.WaitForSingleObject(self._handle,int(timeout*1000))!=0:raise subprocess.TimeoutExpired('guarded child',timeout)
        self.returncode=_winapi.GetExitCodeProcess(self._handle);return self.returncode

    def terminate(self):_winapi.TerminateProcess(self._handle,99)

    def close(self):
        if self._thread:_winapi.CloseHandle(self._thread);self._thread=None
        if self._handle:_winapi.CloseHandle(self._handle);self._handle=None
        if not self.stdin.closed:self.stdin.close()


def verify_worker_membership(name):
    from safe_server_queue import Extended,Rate
    from resource_policy import CPU_PERCENT
    if not name:raise RuntimeError('No named resource job was provided')
    k=C.WinDLL('kernel32',use_last_error=True)
    k.OpenJobObjectW.argtypes=[W.DWORD,W.BOOL,W.LPCWSTR];k.OpenJobObjectW.restype=W.HANDLE
    k.GetCurrentProcess.restype=W.HANDLE
    k.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)];k.IsProcessInJob.restype=W.BOOL
    k.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p];k.QueryInformationJobObject.restype=W.BOOL
    k.CloseHandle.argtypes=[W.HANDLE]
    h=k.OpenJobObjectW(4,False,name)
    if not h:raise C.WinError(C.get_last_error())
    try:
        member=W.BOOL()
        if not k.IsProcessInJob(k.GetCurrentProcess(),h,C.byref(member)) or not member.value:raise RuntimeError('Actual worker escaped resource job')
        limits=Extended();rate=Rate()
        if not k.QueryInformationJobObject(h,9,C.byref(limits),C.sizeof(limits),None):raise C.WinError(C.get_last_error())
        if not k.QueryInformationJobObject(h,15,C.byref(rate),C.sizeof(rate),None):raise C.WinError(C.get_last_error())
        if rate.Flags!=5 or rate.CpuRate>CPU_PERCENT*100 or not(limits.Basic.Flags&0x2000):raise RuntimeError('Worker limits are not the frozen policy')
        return {'actual_worker_pid':os.getpid(),'job_name':name,'in_named_job':True,
            'cpu_hard_cap_percent':rate.CpuRate/100,'job_memory_gib':limits.JobMemory/1024**3}
    finally:
        # Keeping a query handle open would defeat kill-on-last-handle-close.
        k.CloseHandle(h)
