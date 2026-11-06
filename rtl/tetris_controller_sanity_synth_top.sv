`timescale 1ns/1ps

module tetris_controller_sanity_synth_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        lookup_en,
    input  wire [3:0]  lookup_path,
    input  wire [2:0]  lookup_entry,
    output wire        lookup_valid,
    output wire [31:0] lookup_data,
    input  wire        update_en,
    input  wire [3:0]  update_path,
    input  wire [2:0]  update_entry,
    input  wire [31:0] update_data,
    input  wire        desc_push,
    input  wire [31:0] desc_data,
    input  wire        begin_update,
    input  wire        commit_req,
    input  wire        rollback_req,
    output wire        commit_pulse,
    output wire        rollback_pulse,
    output wire        busy,
    output wire [1:0]  hice_state_dbg,
    output wire [4:0]  desc_count,
    output wire [31:0] desc_pop_data
);
    tetris_controller_sanity_top #(
        .PATH_BITS(4),
        .ENTRY_BITS(3),
        .DATA_BITS(32)
    ) u_sanity_top (
        .clk(clk),
        .rst_n(rst_n),
        .lookup_en(lookup_en),
        .lookup_path(lookup_path),
        .lookup_entry(lookup_entry),
        .lookup_valid(lookup_valid),
        .lookup_data(lookup_data),
        .update_en(update_en),
        .update_path(update_path),
        .update_entry(update_entry),
        .update_data(update_data),
        .desc_push(desc_push),
        .desc_data(desc_data),
        .begin_update(begin_update),
        .commit_req(commit_req),
        .rollback_req(rollback_req),
        .commit_pulse(commit_pulse),
        .rollback_pulse(rollback_pulse),
        .busy(busy),
        .hice_state_dbg(hice_state_dbg),
        .desc_count(desc_count),
        .desc_pop_data(desc_pop_data)
    );
endmodule
