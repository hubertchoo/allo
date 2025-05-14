import pickle
import os
import re

def extract_ports(verilog_file):
    """
    Extract the module name and its ports from the Verilog file.
    Assumes the module header only lists the port names, e.g.:
       module gemm_float(a, b, c);
         input [7:0] a;
         output [15:0] b;
         input c;
       endmodule
    This function:
      1. Uses a regex to capture the module name and header port list.
      2. Splits the header list to get port names.
      3. Searches the file for lines declaring port directions and widths.
      4. Returns the module name and a list of tuples: (direction, width, port_name).
    """
    with open(verilog_file, 'r') as f:
        content = f.read()

    # Extract the module header: module <module_name>(port1, port2, ...);
    header_match = re.search(r'module\s+(\w+)\s*\((.*?)\)\s*;', content, re.DOTALL)
    if not header_match:
        # Fallback: use file name (without extension) as module name.
        mod_name = os.path.splitext(os.path.basename(verilog_file))[0]
        print(f"\033[31mWarning: Could not find module header in {verilog_file}; using '{mod_name}' as module name.\033[0m")
        return mod_name, []
    
    mod_name = header_match.group(1)
    port_list_str = header_match.group(2)

    # Get a list of port names (strip whitespace).
    port_names = [p.strip() for p in port_list_str.split(',') if p.strip()]

    # Find port declarations inside the module.
    # We assume declarations are on separate lines like:
    #   input [width] port_name;
    #   output port_name;
    declaration_pattern = re.compile(r'^\s*(input|output|inout)\s*(\[[^\]]+\])?\s*(\w+)\s*;', re.MULTILINE)
    port_declarations = {}
    for m in declaration_pattern.finditer(content):
        direction = m.group(1)
        width = m.group(2) if m.group(2) else ''
        port_name = m.group(3)
        # Only add if this port is in the header list
        if port_name in port_names:
            port_declarations[port_name] = (direction, width)

    # Create a list of ports using the header order.
    ports = []
    for p in port_names:
        if p in port_declarations:
            ports.append((port_declarations[p][0], port_declarations[p][1], p))
        else:
            ports.append(('unknown', '', p))
    return mod_name, ports

def generate_wrapper(verilog_files, dest_verilog_dir):
    """
    Generate a wrapper Verilog file that:
      - Declares a top module named "wrapper" with ports for each
        submodule port (each port name is prefixed by the submodule name).
      - Instantiates each module by mapping its ports to the corresponding
        wrapper ports.
      - Copies the `timescale 1 ns / 1 ps` directive from one of the modules, if available,
        and includes it only once at the top of the wrapper file.
    """
    wrapper_ports = []    # List of tuples: (direction, width, wrapper_port_name)
    instantiations = []   # List of strings with instantiation code
    timescale_line = None

    for verilog_file in verilog_files:
        # Read file content for timescale extraction.
        with open(verilog_file, 'r') as f:
            content = f.read()
        # Capture the timescale directive only once.
        if not timescale_line:
            timescale_match = re.search(r'`timescale\s+1\s+ns\s*/\s*1\s+ps', content)
            if timescale_match:
                timescale_line = timescale_match.group(0)
        
        mod_name, ports = extract_ports(verilog_file)
        
        # For each port in the submodule, create a corresponding wrapper port.
        for direction, width, port_name in ports:
            wrapper_port_name = f"{mod_name}_{port_name}"
            wrapper_ports.append((direction, width, wrapper_port_name))

        # Build the instantiation string for the module.
        inst_lines = []
        inst_lines.append(f"  {mod_name} u_{mod_name} (")
        port_connections = []
        for direction, width, port_name in ports:
            wrapper_port_name = f"{mod_name}_{port_name}"
            port_connections.append(f"    .{port_name}({wrapper_port_name})")
        inst_lines.append(",\n".join(port_connections))
        inst_lines.append("  );")
        instantiations.append("\n".join(inst_lines))

    # Build the wrapper module code.
    lines = []
    # Include the timescale directive only once at the top if it was found.
    if timescale_line:
        lines.append(timescale_line)
    lines.append("module wrapper(")
    # List all wrapper ports in the module header.
    port_declarations = []
    for direction, width, name in wrapper_ports:
        port_declarations.append(f"  {name}")
    lines.append(",\n".join(port_declarations))
    lines.append(");")
    lines.append("")
    # Declare the port directions and widths inside the wrapper.
    for direction, width, name in wrapper_ports:
        decl = f"{direction} {width} {name};" if width else f"{direction} {name};"
        lines.append(decl)
    lines.append("")
    # Add each submodule instantiation.
    for inst in instantiations:
        lines.append(inst)
        lines.append("")
    lines.append("endmodule")
    
    wrapper_code = "\n".join(lines)
    
    # Write the wrapper file into the destination directory.
    dest_file = os.path.join(dest_verilog_dir, "wrapper.v")
    with open(dest_file, 'w') as f:
        f.write(wrapper_code)
    
    print(f"Wrapper Verilog file created at \033[33m{dest_file}\033[0m")
