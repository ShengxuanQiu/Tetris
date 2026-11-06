set script_dir [file normalize [file dirname [info script]]]
set build_dir [file join $script_dir build vivado]
file mkdir $build_dir

set srcs [list \
    [file join $script_dir tetris_hatt.sv] \
    [file join $script_dir tetris_desc_queue.sv] \
    [file join $script_dir tetris_hice_ctrl.sv] \
    [file join $script_dir tetris_controller_sanity_top.sv] \
    [file join $script_dir tb_tetris_controller_sanity.sv] \
]

set sim_dir [file join $build_dir sim]
file mkdir $sim_dir
cd $sim_dir

puts "INFO: Running Vivado xsim behavioral RTL sanity simulation"
exec xvlog -sv {*}$srcs >@ stdout 2>@ stderr
exec xelab -debug typical tb_tetris_controller_sanity -s tetris_controller_sanity >@ stdout 2>@ stderr
exec xsim tetris_controller_sanity -runall -log [file join $sim_dir xsim_run.log] >@ stdout 2>@ stderr

set sim_log [file join $sim_dir xsim_run.log]
set sim_text [read [open $sim_log r]]
if {[string first "PASS: minimal Tetris controller-side RTL sanity test" $sim_text] < 0} {
    error "RTL simulation did not report PASS. See $sim_log"
}
puts "INFO: RTL behavioral simulation PASS"

set synth_dir [file join $build_dir synth]
file delete -force $synth_dir
file mkdir $synth_dir
cd $synth_dir

puts "INFO: Running Vivado out-of-context synthesis sanity check"
create_project -force tetris_controller_sanity_ooc $synth_dir -part xc7a35tcpg236-1
set synth_srcs [list \
    [file join $script_dir tetris_hatt.sv] \
    [file join $script_dir tetris_desc_queue.sv] \
    [file join $script_dir tetris_hice_ctrl.sv] \
    [file join $script_dir tetris_controller_sanity_top.sv] \
    [file join $script_dir tetris_controller_sanity_synth_top.sv] \
]
add_files $synth_srcs
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1
synth_design -top tetris_controller_sanity_synth_top -mode out_of_context -flatten_hierarchy rebuilt
create_clock -period 5.000 -name clk [get_ports clk]
report_utilization -file [file join $synth_dir utilization_synth.rpt]
report_timing_summary -file [file join $synth_dir timing_synth.rpt]
write_checkpoint -force [file join $synth_dir tetris_controller_sanity_ooc.dcp]

set status [open [file join $build_dir rtl_sanity_summary.txt] w]
puts $status "Vivado version: [version -short]"
puts $status "Behavioral simulation: PASS"
puts $status "Simulation log: [file normalize $sim_log]"
puts $status "Waveform VCD: [file normalize [file join $sim_dir tetris_controller_sanity.vcd]]"
puts $status "Out-of-context synthesis: PASS (small-parameter sanity wrapper)"
puts $status "Utilization report: [file normalize [file join $synth_dir utilization_synth.rpt]]"
puts $status "Timing report: [file normalize [file join $synth_dir timing_synth.rpt]]"
puts $status "Scope: controller-local HATT metadata, descriptor queue, and HICE commit/rollback FSM sanity only."
close $status

puts "INFO: Vivado RTL sanity flow PASS"
