`timescale 1ns/1ps

module tetris_controller_sanity_top #(
    parameter PATH_BITS = 8,
    parameter ENTRY_BITS = 4,
    parameter DATA_BITS = 64
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         lookup_en,
    input  wire [PATH_BITS-1:0]         lookup_path,
    input  wire [ENTRY_BITS-1:0]        lookup_entry,
    output wire                         lookup_valid,
    output wire [DATA_BITS-1:0]         lookup_data,
    input  wire                         update_en,
    input  wire [PATH_BITS-1:0]         update_path,
    input  wire [ENTRY_BITS-1:0]        update_entry,
    input  wire [DATA_BITS-1:0]         update_data,
    input  wire                         desc_push,
    input  wire [DATA_BITS-1:0]         desc_data,
    input  wire                         begin_update,
    input  wire                         commit_req,
    input  wire                         rollback_req,
    output wire                         commit_pulse,
    output wire                         rollback_pulse,
    output wire                         busy,
    output wire [1:0]                   hice_state_dbg,
    output wire [4:0]                   desc_count,
    output wire [DATA_BITS-1:0]         desc_pop_data
);
    wire queue_empty;
    wire queue_full;
    wire queue_pop;

    tetris_hatt #(
        .PATH_BITS(PATH_BITS),
        .ENTRY_BITS(ENTRY_BITS),
        .DATA_BITS(DATA_BITS)
    ) u_hatt (
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
        .update_data(update_data)
    );

    tetris_desc_queue #(
        .DEPTH(8),
        .DATA_BITS(DATA_BITS),
        .PTR_BITS(4)
    ) u_desc_queue (
        .clk(clk),
        .rst_n(rst_n),
        .push(desc_push && !queue_full),
        .push_data(desc_data),
        .pop(queue_pop),
        .pop_data(desc_pop_data),
        .empty(queue_empty),
        .full(queue_full),
        .count(desc_count)
    );

    tetris_hice_ctrl u_hice (
        .clk(clk),
        .rst_n(rst_n),
        .begin_update(begin_update),
        .commit_req(commit_req),
        .rollback_req(rollback_req),
        .queue_empty(queue_empty),
        .queue_pop(queue_pop),
        .commit_pulse(commit_pulse),
        .rollback_pulse(rollback_pulse),
        .busy(busy),
        .state_dbg(hice_state_dbg)
    );
endmodule
