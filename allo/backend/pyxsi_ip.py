import pyxsi
import struct
import numpy as np
from abc import ABC, abstractmethod

HALF_PERIOD = 5000

def int32_to_binary(n):
    """Convert an int32 number to its 32-bit binary representation."""
    return format(n, '032b')  # Ensure it's treated as a 32-bit integer

def binary_to_int32(binary_str):
    """Convert a 32-bit binary string to an int32 integer."""
    num = int(binary_str, 2)  # Convert binary to integer
    if num >= 2**31:  # Handle two's complement for negative numbers
        num -= 2**32
    return num

def float32_to_ieee754(f):
    """Convert a float32 to its 32-bit IEEE 754 binary representation."""
    packed = struct.pack('!f', f)  # Convert to 4-byte binary
    int_rep = struct.unpack('!I', packed)[0]  # Interpret as 32-bit integer
    return format(int_rep, '032b')  # Convert to binary string

def ieee754_to_float32(binary_str):
    """Convert a 32-bit IEEE 754 binary string to a float32."""
    int_rep = int(binary_str, 2)  # Convert binary to int
    packed = struct.pack('!I', int_rep)  # Pack int as bytes
    return struct.unpack('!f', packed)[0]  # Convert bytes to float

class PythonRTLArgumentInterface(ABC):
    # An abstract base class (ABC) cannot be instantiated on its own.
    # It can only serve as a base class for another class.
    @abstractmethod
    def sync_interface(self):
        """This abstractmethod MUST be overridden by any subclass."""
        pass

class ArrayApMemInterface(PythonRTLArgumentInterface):
    def __init__(
        self, pyxsi_sim, arg_name, np_array_obj, dtype
    ):
        self.arg_name = arg_name
        self.np_array_obj = np_array_obj.ravel()
        self.sim = pyxsi_sim
        self.dtype = dtype
        
        self.port_name_list = []
        for i in range(self.sim.get_port_count()):
            self.port_name_list.append(self.sim.get_port_name(i))
            
        if f"{self.arg_name}_we0" in self.port_name_list:
            self.direction = "MemWrite"
        else:
            self.direction = "MemRead"
        self.next_cycle_read = False
        self.next_cycle_read_addr = None

    def sync_interface(self):
        # Returns reads triggered by chip_enable on previous cycle
        if self.direction == "MemRead" and self.next_cycle_read:
            
            # setattr(self.port_name_list, f"{self.arg_name}_q0", self.np_array_obj[self.next_cycle_read_addr])     
            if self.dtype == "int32": # int32
                self.sim.set_port_value(f"{self.arg_name}_q0", f"{int32_to_binary(self.np_array_obj[self.next_cycle_read_addr])}")
            else: # float32 (ieee754)
                self.sim.set_port_value(f"{self.arg_name}_q0", f"{float32_to_ieee754(self.np_array_obj[self.next_cycle_read_addr])}")
            self.next_cycle_read = False
            self.next_cycle_read_addr = None

        # array_addr = getattr(self.port_name_list, f"{self.arg_name}_address0")
        array_addr = self.sim.get_port_value(f"{self.arg_name}_address0")
        array_addr = int(array_addr, 2) if "X" not in array_addr else 0
        
        # chip_en = getattr(self.port_name_list, f"{self.arg_name}_ce0")
        chip_en = self.sim.get_port_value(f"{self.arg_name}_ce0")
        chip_en = int(chip_en, 2) if "X" not in chip_en else 0
        
        if self.direction == "MemRead":
            # Write from numpy array into Verilator model
            if chip_en:
                # Read data after one cycle
                self.next_cycle_read = True
                self.next_cycle_read_addr = array_addr
        elif self.direction == "MemWrite":
            # Read out of Verilator model into numpy array
            write_en = self.sim.get_port_value(f"{self.arg_name}_we0")
            write_en = int(write_en, 2) if "X" not in write_en else 0
            
            if chip_en and write_en:
                # Write data on some cycle
                temp = self.sim.get_port_value(f"{self.arg_name}_d0")
                if self.dtype == "int32": # int32
                    temp = binary_to_int32(temp) if "X" not in temp else 0
                else: # float32 (ieee754)
                    temp = ieee754_to_float32(temp) if "X" not in temp else 0
                self.np_array_obj[array_addr] = temp

class PyxsiIPModule:
    def __init__(
        self, top_func_name, pyxsi_sim, signature, dtype
    ):
        self.top_func_name = top_func_name
        self.sim = pyxsi_sim
        self.signature = signature
        self.dtype = dtype
        self.interface_map = {}

    def parse_args(self, args):
        assert (len(args) == len(self.signature)), "Number of Python arguments do not match number of HLS arguments."
        for arg, sig, dtype in zip(args, self.signature, self.dtype):
            if isinstance(arg, np.ndarray):
                assert ("[" in sig), f"Python argument type does not match HLS argument type."
                hls_arg_name = sig.split(" ")[0]
                self.interface_map[hls_arg_name] = ArrayApMemInterface(self.sim, hls_arg_name, arg, dtype)
            # FUTURE: Add support for other argument types here
            
    def clk_tick(self):
        self.sim.set_port_value("ap_clk", "1")
        self.sim.run(HALF_PERIOD)
        self.sim.set_port_value("ap_clk", "0")
        self.sim.run(HALF_PERIOD)

    def reset_module(self):
        self.sim.set_port_value("ap_rst", "1")
        self.clk_tick()
        self.sim.set_port_value("ap_rst", "0")
        self.clk_tick()
        self.sim.set_port_value("ap_start", "1")

    def run_module(self):
        self.reset_module()
        while self.sim.get_port_value("ap_done") != "1":
            for intf in self.interface_map.values():
                intf.sync_interface()
            self.clk_tick()
        self.sim.set_port_value("ap_start", "0")

    def __call__(self, *args):
        self.parse_args(args)
        self.run_module()