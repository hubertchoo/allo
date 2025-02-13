import numpy as np
from abc import ABC, abstractmethod

ip_collection_mode = False

class PythonRTLArgumentInterface(ABC):
    # An abstract base class (ABC) cannot be instantiated on its own.
    # It can only serve as a base class for another class.
    @abstractmethod
    def sync_interface(self):
        """This abstractmethod MUST be overridden by any subclass."""
        pass

class ArrayApMemInterface(PythonRTLArgumentInterface):
    def __init__(
        self, pyverilator_sim, arg_name, np_array_obj
    ):
        self.arg_name = arg_name
        self.np_array_obj = np_array_obj.ravel()
        self.sim = pyverilator_sim
        if f"{self.arg_name}_we0" in self.sim.io:
            self.direction = "MemWrite"
        else:
            self.direction = "MemRead"
        self.next_cycle_read = False
        self.next_cycle_read_addr = None

    def sync_interface(self):
        # Returns reads triggered by chip_enable on previous cycle
        if self.direction == "MemRead" and self.next_cycle_read:
            setattr(self.sim.io, f"{self.arg_name}_q0", self.np_array_obj[self.next_cycle_read_addr])
            self.next_cycle_read = False
            self.next_cycle_read_addr = None

        array_addr = getattr(self.sim.io, f"{self.arg_name}_address0")
        chip_en = getattr(self.sim.io, f"{self.arg_name}_ce0")
        if self.direction == "MemRead":
            # Write from numpy array into Verilator model
            if chip_en:
                # Read data after one cycle
                self.next_cycle_read = True
                self.next_cycle_read_addr = array_addr
        elif self.direction == "MemWrite":
            # Read out of Verilator model into numpy array
            write_en = getattr(self.sim.io, f"{self.arg_name}_we0")
            if chip_en and write_en:
                # Write data on some cycle
                self.np_array_obj[array_addr] = getattr(self.sim.io, f"{self.arg_name}_d0")

class PyverilatorIPModule:
    def __init__(
        self, top_func_name, pyverilator_sim, signature
    ):
        self.top_func_name = top_func_name
        self.sim = pyverilator_sim
        self.signature = signature
        self.interface_map = {}
        self.call_returns_self = False

    def parse_args(self, args):
        assert (len(args) == len(self.signature)), "Number of Python arguments do not match number of HLS arguments."
        for arg, sig in zip(args, self.signature):
            if isinstance(arg, np.ndarray):
                assert ("[" in sig), f"Python argument type does not match HLS argument type."
                hls_arg_name = sig.split(" ")[0]
                self.interface_map[hls_arg_name] = ArrayApMemInterface(self.sim, hls_arg_name, arg)
            # FUTURE: Add support for other argument types here

    def reset_module(self):
        self.sim.io.ap_rst = 1
        self.sim.clock.tick()
        self.sim.io.ap_rst = 0
        self.sim.clock.tick()

    def start_module(self):
        self.sim.io.ap_start = 1

    def sync_module(self):
        for intf in self.interface_map.values():
            intf.sync_interface()

    def run_module(self):
        self.sim.start_vcd_trace(f"{self.top_func_name}.vcd")
        self.reset_module()
        self.start_module()
        while not self.sim.io.ap_done:
            self.sync_module()
            self.sim.clock.tick()
        self.sim.io.ap_start = 0


    def __call__(self, *args):
        self.parse_args(args)
        if not ip_collection_mode:
            self.run_module()

class ParallelIPModuleCollection:
    def __init__(self, *pyverilator_ip_modules):
        self.pyverilator_ip_modules = list(pyverilator_ip_modules)

    def reset_all_modules(self):
        for ip_module in self.pyverilator_ip_modules:
            ip_module.reset_module()

    def start_all_modules(self):
        for ip_module in self.pyverilator_ip_modules:
            ip_module.start_module()

    def run_all_modules(self):
        def check_all_modules_done():
            for ip_module in self.pyverilator_ip_modules:
                if not ip_module.sim.io.ap_done:
                    return False
            return True

        self.reset_all_modules()
        self.start_all_modules()
        while not check_all_modules_done():
            # does each signal keep staying done HIGH, or is it only done HIGH for one cycle?
            for ip_module in self.pyverilator_ip_modules:
                ip_module.sync_module()
                ip_module.sim.clock.tick()
        for ip_module in self.pyverilator_ip_modules:
            ip_module.sim.io.ap_start = 0

    def __call__(self, *args):
        self.run_all_modules()

class SequentialIPModuleCollection:
    def __init__(self, *pyverilator_ip_modules):
        self.pyverilator_ip_modules = list(pyverilator_ip_modules)

    def reset_all_modules(self):
        for ip_module in self.pyverilator_ip_modules:
            ip_module.reset_module()

    def run_all_modules(self):
        self.reset_all_modules()

        for curr_ip_module in self.pyverilator_ip_modules:
            curr_ip_module.start_module()

            while not curr_ip_module.sim.io.ap_done:
                # Sync and tick for all modules
                for ip_module in self.pyverilator_ip_modules:
                    ip_module.sync_module()
                    ip_module.sim.clock.tick()

            curr_ip_module.sim.io.ap_start = 0

    def __call__(self, *args):
        self.run_all_modules()

class IPCollectionModeContext:
    def __enter__(self):
        global ip_collection_mode
        ip_collection_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global ip_collection_mode
        ip_collection_mode = False