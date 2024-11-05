# generate_ip_sim.tcl template
   
# Create a Vivado project
create_project -force ip_out ./ip_out
   
# Directory containing IP generation scripts
set script_dir "./out.prj/solution1/syn/verilog"

# Source all .tcl files in the directory
foreach file [glob -nocomplain -directory $script_dir *.tcl] {
    puts "Sourcing $file"
    source $file
}
   
# Add Verilog files to the project
add_files -norecurse [glob -nocomplain -directory $script_dir *.v]
   
# Set the top module name
# Replace "top_module_name" with the actual name
set_property top top_module_name [current_fileset -simset]
   
# Launch simulation
launch_simulation -help
set_property target_simulator "XSim" [current_project]
launch_simulation

# Export simulation
export_simulation \
          -force \
          -simulator xsim \
          -directory test_xsim \
          -lib_map_path test_xsim \
          -use_ip_compiled_libs
   
# Exit the project
close_project
