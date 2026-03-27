import subprocess
import json
import select
import psutil
import os
import time
from typing import Optional


class LeanServer:

  def __init__(self, path_to_repl: str, verbose: bool=False, import_timeout: float=30.0, env_variables: Optional[dict]=None) -> None:
    """
    Args:
      path_to_repl: the path to the repl, which is the directory of the repl
      verbose: whether to print the verbose information
      import_timeout: the timeout for importing Mathlib
      env_variables: the environment variables for the repl
    """

    self.verbose = verbose
    self.path_to_repl = path_to_repl
    self.import_timeout = import_timeout
    self.env_variables = env_variables if env_variables is not None else os.environ.copy()
    self.repl_initialize()

  def run_sketch(self, sketch: str, timeout: int=5, auto_check_and_reinitialize: bool=True, memory_limit: float=None) -> dict:

    """
    Args:
      sketch: the sketch to run, should not include `import`
      timeout: the timeout for the sketch
      auto_check_and_reinitialize: whether to check and reinitialize the repl if it is not healthy
      memory_limit: the memory limit(MB) for the repl, if the memory limit is exceeded, the repl will be terminated, and the error message will be returned

    Returns:
      dict: The output of the sketch, if the sketch is not stopped normally, the output will be None
    """
    if auto_check_and_reinitialize:
      if self.is_not_healthy():
        self.repl_initialize()

    msg, signal = self._send_and_receive(sketch, env=0, timeout=timeout, memory_limit=memory_limit)

    if signal == 1:
      output = json.loads(msg)
      output.pop("env")
    elif signal == 2:
      output = dict(repl_err=msg)
    else:
      if self.verbose:
        print(".run_sketch: Time out! terminate the process")
      output = None

    return output
    
  def _send_and_receive(self, sketch: str, env: int=None, timeout: int=30, memory_limit: float=None) -> str:
    assert self.process.poll() is None, "Lean server has been died!"

    if env is None:
      command = json.dumps(dict(cmd=sketch), ensure_ascii=False)
    else:
      command = json.dumps(dict(cmd=sketch, env=env), ensure_ascii=False)

    self.process.stdin.write(command + "\n")
    self.process.stdin.flush()
    self.process.stdin.write("\n")
    self.process.stdin.flush()

    end_time = time.time() + timeout

    while time.time() < end_time:
      # 检查内存使用情况
      if memory_limit is not None:
          memory_info = self.get_memory_info()
          if memory_info and memory_info['repl'] and isinstance(memory_info['repl'], dict):
              repl_memory = memory_info['repl']['rss']
              if repl_memory > memory_limit:
                  memory_error = f"Memory usage ({repl_memory:.2f}MB) exceeded limit ({memory_limit}MB)"
                  if self.verbose:
                      print(f"._send_and_receive: repl error: {memory_error}")
                  self._close() # 这里得立即关闭，避免炸内存
                  return memory_error, 2

      readable, _, _ = select.select([self.process.stdout], [], [], 0.5)

      if readable:
        last_line = self.process.stdout.readline()
        if not self.is_alive() or last_line == "":
          stderr_line = self.process.stderr.readline()
          if self.verbose:
            print(f"._send_and_receive: repl error: {stderr_line}")
          self._close() # 这里得立即关闭
          return stderr_line, 2

        msg = ""
        while last_line != "\n":
          if self.verbose:
            print(f"._send_and_receive: last_line: {last_line}")
          msg += last_line
          last_line = self.process.stdout.readline()
        return msg, 1

    if self.verbose:
      print(f"._send_and_receive: Lean server has not responded in {timeout} seconds")

    return None, 0
  
  def is_alive(self) -> bool:
    """ This function is used to check if the Lean server is running.
    """
    return self.process.poll() is None
  
  def is_not_healthy(self) -> int:
    """ This function is used to check if the Lean server is not healthy.
    """
    if not self.is_alive():
      if self.verbose:
        print("is_not_healthy: Lean server is not running")
      return 1
    
    readable, _, _ = select.select([self.process.stdout], [], [], 0.5)
    
    if readable:
      if self.verbose:
        print("is_not_healthy: Something in stdout has not been read")
      return 2
    
    _, signal = self._send_and_receive("def x := 0", env=0, timeout=0.5)

    if signal != 1:
      if self.verbose:
        print("is_not_healthy: Repl does not respond normally")
      return 3
    
    return 0

  def _close(self) -> None:
    """ This function is used to close the Lean server, terminate the process, and free memory.
    """
    if hasattr(self, 'process'):
      if self.process.stdin and not self.process.stdin.closed:
        self.process.stdin.flush()
        self.process.stdin.close()
      if self.process.stdout and not self.process.stdout.closed:
        self.process.stdout.flush()
        self.process.stdout.close()
      if self.process.stderr and not self.process.stderr.closed:
        self.process.stderr.flush()
        self.process.stderr.close()

      # Must terminate the repl process before terminating the lake process
      if self.is_alive():
        lake_pid = self.process.pid
        lake_process = psutil.Process(lake_pid)
        children = lake_process.children(recursive=True) # get the children of the lake process on time
        for child in children:
          if child.is_running():
            child.terminate()
            child.wait(timeout=5)
            if psutil.pid_exists(child.pid):
              child.kill()
              child.wait(timeout=5)
          
          if self.verbose:
            print(f"._close: child.pid: {child.pid}, psutil.pid_exists(child.pid): {psutil.pid_exists(child.pid)}")
            
        if lake_process.is_running():
          lake_process.terminate()
          lake_process.wait(timeout=5)
          if psutil.pid_exists(lake_pid):
            lake_process.kill()
            lake_process.wait(timeout=5)

        if self.verbose:
          print(f"._close: psutil.pid_exists(lake_pid): {psutil.pid_exists(lake_pid)}")

  def repl_initialize(self, timeout: Optional[int]=None) -> int:
    self._close()

    self.process = subprocess.Popen(
      ["lake", "exe", "repl"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, 
      encoding="utf-8",
      cwd=self.path_to_repl,
      env=self.env_variables
    )

    if not self.is_alive():
      if self.verbose:
        print("repl_initialize: Lean server cannot start normally")
      raise RuntimeError("Lean server cannot started normally")

    if timeout is None:
      timeout = self.import_timeout
    msg, signal = self._send_and_receive("import Mathlib", timeout=timeout)

    if signal != 1:
      if self.verbose:
        print(f"repl_initialize: Lean server cannot import Mathlib in {timeout} seconds")
      self._close()
      raise ImportError(f"Lean server cannot import Mathlib in {timeout} seconds")
    
    elif msg != '{"env": 0}\n':
      if self.verbose:
        print(f"repl_initialize: Unexpected output when importing Mathlib: {msg}")
      self._close()
      raise ImportError(f"Unexpected output when importing Mathlib: {msg}")
    else:  
      return 0
    
  def _kill(self) -> None:
    """ This function is used to kill the Lean server, ready to deprecated.
    """
    self._close()
    
  def get_memory_info(self) -> dict:
    """ This function is used to get the memory information of the Lean server.
    The self.process.pid is not the actual lean repl process, its name is `lake`, its memory usage will be constant. The actual lean repl process is the child process of `lake`, named as `repl`, the memeory of `repl` will be nondecreasing because lean is lack of garbage collection, so it may lead to OOM if the server is running for a long time. This function is used to watch the memory usage of the actual lean repl process. We highly suggest you to restart the server if the memory increase is too much.

    Returns:
      dict: The memory information of the Lean server
    """

    if not self.is_alive():
      return None
    
    pid = self.process.pid
    try:
      process = psutil.Process(pid)
      memory_info = process.memory_info()
      memory_full = process.memory_full_info()

      # RSS (Resident Set Size): 进程在物理内存中占用的空间，包括共享库
      rss = memory_info.rss/1024/1024
      # VMS (Virtual Memory Size): 进程可访问的所有内存，包括未实际使用的
      vms = memory_info.vms/1024/1024
      # USS (Unique Set Size): 进程独占的物理内存，不包括共享库
      # USS是最能真实反映进程实际占用内存的指标，因为它只计算进程独占的内存
      uss = getattr(memory_full, 'uss', 0)/1024/1024

      lake_info = {
        "pid": process.pid,
        "cmdline": process.cmdline(),
        "rss": rss,
        "vms": vms,
        "uss": uss,
      }
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
      return None

    # 获取子进程信息
    children = process.children(recursive=True)
    repl_info = None
    for child in children:
      try:
        child_memory_info = child.memory_info()
        child_memory_full = child.memory_full_info()
        child_rss = child_memory_info.rss/1024/1024
        child_vms = child_memory_info.vms/1024/1024
        child_uss = getattr(child_memory_full, 'uss', 0)/1024/1024
        repl_info = {
          "pid": child.pid,
          "cmdline": child.cmdline(),
          "rss": child_rss,
          "vms": child_vms,
          "uss": child_uss,
        }
        break
      except (psutil.NoSuchProcess, psutil.ZombieProcess) as e:
        repl_info = str(e)
        continue
      
    if self.verbose:
      print(f"get_memory_info: lake pid: {pid}, cmdline: {lake_info['cmdline']}, RSS: {lake_info['rss']:.2f} MB, VMS: {lake_info['vms']:.2f} MB, USS: {lake_info['uss']:.2f} MB")

      if repl_info is not None and isinstance(repl_info, dict):
        print(f"get_memory_info: repl pid: {repl_info['pid']}, cmdline: {repl_info['cmdline']}, RSS: {repl_info['rss']:.2f} MB, VMS: {repl_info['vms']:.2f} MB, USS: {repl_info['uss']:.2f} MB")

    return {
      "lake": lake_info,
      "repl": repl_info,
    }
