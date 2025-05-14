import struct
import numpy as np
from abc import ABC, abstractmethod

HALF_PERIOD = 5000

done_num = 0

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
        self, pyxsi_sim, arg_name, np_array_obj, dtype, mod_name
    ):
        self.arg_name = arg_name
        self.np_array_obj = np_array_obj.ravel()
        self.sim = pyxsi_sim
        self.dtype = dtype
        self.mod_name = mod_name
        
        self.port_name_list = []
        for i in range(self.sim.get_port_count()):
            self.port_name_list.append(self.sim.get_port_name(i))
            
        if f"{self.mod_name}_{self.arg_name}_we0" in self.port_name_list:
            self.direction = "MemWrite"
        else:
            self.direction = "MemRead"
        self.next_cycle_read = False
        self.next_cycle_read_addr = None
        
        print(f"{self.mod_name:<12}  {self.arg_name:<6}  {self.dtype:<12}  {self.direction:<12}  {len(self.np_array_obj):<8}  {self.np_array_obj}")

    def sync_interface(self):
        # Returns reads triggered by chip_enable on previous cycle
        if self.direction == "MemRead" and self.next_cycle_read:
             
            if self.dtype == "int32": # int32
                self.sim.set_port_value(f"{self.mod_name}_{self.arg_name}_q0", f"{int32_to_binary(self.np_array_obj[self.next_cycle_read_addr])}")
            else: # float32 (ieee754)
                self.sim.set_port_value(f"{self.mod_name}_{self.arg_name}_q0", f"{float32_to_ieee754(self.np_array_obj[self.next_cycle_read_addr])}")
            self.next_cycle_read = False
            self.next_cycle_read_addr = None

        array_addr = self.sim.get_port_value(f"{self.mod_name}_{self.arg_name}_address0")
        array_addr = int(array_addr, 2) if "X" not in array_addr else 0
        
        chip_en = self.sim.get_port_value(f"{self.mod_name}_{self.arg_name}_ce0")
        chip_en = int(chip_en, 2) if "X" not in chip_en else 0
        
        if self.direction == "MemRead":
            # Write from numpy array into xsi model
            if chip_en:
                # Read data after one cycle
                self.next_cycle_read = True
                self.next_cycle_read_addr = array_addr
        elif self.direction == "MemWrite":
            # Read out of xsi model into numpy array
            write_en = self.sim.get_port_value(f"{self.mod_name}_{self.arg_name}_we0")
            write_en = int(write_en, 2) if "X" not in write_en else 0
            
            if chip_en and write_en:
                # Write data on some cycle
                temp = self.sim.get_port_value(f"{self.mod_name}_{self.arg_name}_d0")
                if self.dtype == "int32": # int32
                    temp = binary_to_int32(temp) if "X" not in temp else 0
                else: # float32 (ieee754)
                    temp = ieee754_to_float32(temp) if "X" not in temp else 0
                self.np_array_obj[array_addr] = temp
                
                # print(f"write_en - {self.mod_name} - Address: {array_addr} - Data: {temp}")

class PyxsiIPModule:
    def __init__(
        self, top_func_name, pyxsi_sim, signature, dtype, mods
    ):
        self.top_func_name = top_func_name
        self.sim = pyxsi_sim
        self.signature = signature
        self.dtype = dtype
        self.mods = mods
        self.interface_map = {} # dictionary of dictionary "intf_map" using "mod_name" as the key
        self.call_returns_self = False

    def parse_args(self, args):
        for sub_args, sub_signature in zip(args, self.signature):
            assert (len(sub_args) == len(sub_signature)), "Number of Python arguments do not match number of HLS arguments."
        for sub_args, sub_signature, sub_dtype, mod_name in zip(args, self.signature, self.dtype, self.mods):
            intf_map = {}
            for arg, sig, dtype in zip(sub_args, sub_signature, sub_dtype):
                if isinstance(arg, np.ndarray):
                    assert ("[" in sig), f"Python argument type does not match HLS argument type."
                    hls_arg_name = sig.split(" ")[0]
                    intf_map[hls_arg_name] = ArrayApMemInterface(self.sim, hls_arg_name, arg, dtype, mod_name)
                    # FUTURE: Add support for other argument types here
            self.interface_map[mod_name] = intf_map
        print(f"interface_map: {self.interface_map}")
            
    def clk_tick(self, mod_name):
        self.sim.set_port_value(mod_name + "_ap_clk", "1")
        self.sim.run(HALF_PERIOD)
        self.sim.set_port_value(mod_name + "_ap_clk", "0")
        self.sim.run(HALF_PERIOD)

    def reset_module(self, mod_name):
        self.sim.set_port_value(mod_name + "_ap_rst", "1")
        self.clk_tick(mod_name)
        self.sim.set_port_value(mod_name + "_ap_rst", "0")
        self.clk_tick(mod_name)
        
    def start_module(self, mod_name):
        self.sim.set_port_value(mod_name + "_ap_start", "1")
        self.clk_tick(mod_name)
        
    def sync_module(self, mod_name):
        for intf in self.interface_map[mod_name].values():
            intf.sync_interface()

    def __call__(self, args):
        self.parse_args(args)
            
class ParallelIPModuleCollection:
    def __init__(self, *pyxsi_ip_modules):
        self.ip_module = pyxsi_ip_modules[-1]
        
        #  About self.ip_module:
        #
        #  mod = PyxsiIPModule(
        #    top_func_name = "wrapper",
        #    pyxsi_sim = pyxsi.XSI( fr"./xsim.dir/wrapper_behav/xsimk.so", language=pyxsi.VERILOG ),
        #    signature = [ [ mod1_sig1, mod1_sig2, mod1_sig3 ], [ mod2_sig1, mod2_sig2, mod2_sig3 ] ],
        #    dtype = [ [ mod1_dtype1, mod1_dtype2, mod1_dtype3 ], [ mod2_dtype1, mod2_dtype2, mod2_dtype3] ],
        #    mods = [ mod1, mod2 ],
        #  )
        #
        #  mod( [ [ mod1_argdata1, mod1_argdata2, mod1_argdata3 ], [ mod2_argdata1, mod2_argdata2, mod2_argdata3 ] ] )
        
    def reset_all_modules(self):
        for mod_name in self.ip_module.mods:
            self.ip_module.reset_module(mod_name)
            
    def start_all_modules(self):
        for mod_name in self.ip_module.mods:
            self.ip_module.start_module(mod_name)
            
    def run_all_modules(self):
        def check_all_modules_done():
            for mod_name in self.ip_module.mods:
                global done_num
                if self.ip_module.sim.get_port_value(mod_name + "_ap_done") == "1":
                    done_num += 1
                    self.ip_module.sim.set_port_value(mod_name + "_ap_start", "0")
                    
            if done_num == len(self.ip_module.mods):
                return True
            else:
                return False
        
        self.reset_all_modules()
        self.start_all_modules()
        while not check_all_modules_done():
            for mod_name in self.ip_module.mods:
                self.ip_module.sync_module(mod_name)
                self.ip_module.clk_tick(mod_name)
                
        np.testing.assert_allclose(self.ip_module.interface_map["gemm_float"]["v17"].np_array_obj.reshape((32, 32)), 
                                   np.dot(self.ip_module.interface_map["gemm_float"]["v15"].np_array_obj.reshape((32, 32)), 
                                          self.ip_module.interface_map["gemm_float"]["v16"].np_array_obj.reshape((32, 32))), rtol=1e-6, atol=1e-3)
        
        np.testing.assert_allclose(self.ip_module.interface_map["gemm_int"]["v17"].np_array_obj.reshape((32, 32)), 
                                   np.dot(self.ip_module.interface_map["gemm_int"]["v15"].np_array_obj.reshape((32, 32)), 
                                          self.ip_module.interface_map["gemm_int"]["v16"].np_array_obj.reshape((32, 32))), rtol=1e-6, atol=1e-3)
        
        np.testing.assert_allclose(self.ip_module.interface_map["gemm_mix"]["v17"].np_array_obj.reshape((32, 32)), 
                                   np.dot(self.ip_module.interface_map["gemm_mix"]["v15"].np_array_obj.reshape((32, 32)), 
                                          self.ip_module.interface_map["gemm_mix"]["v16"].np_array_obj.reshape((32, 32))), rtol=1e-6, atol=1e-3)
                
        print(f"{self.ip_module.interface_map["gemm_float"]["v17"].np_array_obj}")
        print(f"{self.ip_module.interface_map["gemm_int"]["v17"].np_array_obj}")
        print(f"{self.ip_module.interface_map["gemm_mix"]["v17"].np_array_obj}")
        
        print(f"\nPARALLEL TEST PASSED\n")
            
    def __call__(self):
        self.run_all_modules()

# class SequentialIPModuleCollection:
#     def __init__(self, *pyxsi_ip_modules):
#         self.ip_module = pyxsi_ip_modules[-1]
#         print(pyxsi_ip_modules[-1])
        
#     def reset_all_modules(self):
#         for mod_name in self.ip_module.mods:
#             self.ip_module.reset_module(mod_name)
            
#     def run_all_modules(self):
#         self.reset_all_modules()
        
#         for mod_name in self.ip_module.mods:
#             self.ip_module.start_module(mod_name)
            
#             while self.ip_module.sim.get_port_value(mod_name + "_ap_done") != "1":
#                 # Sync and tick for all modules
#                 for mod_name in self.ip_module.mods:
#                     self.ip_module.sync_module(mod_name)
#                     self.ip_module.clk_tick(mod_name)
                    
#             self.ip_module.sim.set_port_value(mod_name + "_ap_start", "0")
            
#     def __call__(self, *args):
#         self.run_all_modules()
