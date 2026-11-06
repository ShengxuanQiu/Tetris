`timescale 1ns/1ps

module tetris_hatt #(
    parameter PATH_BITS = 8,
    parameter ENTRY_BITS = 4,
    parameter DATA_BITS = 64
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         lookup_en,
    input  wire [PATH_BITS-1:0]         lookup_path,
    input  wire [ENTRY_BITS-1:0]        lookup_entry,
    output reg                          lookup_valid,
    output reg  [DATA_BITS-1:0]         lookup_data,
    input  wire                         update_en,
    input  wire [PATH_BITS-1:0]         update_path,
    input  wire [ENTRY_BITS-1:0]        update_entry,
    input  wire [DATA_BITS-1:0]         update_data
);
    localparam DEPTH = (1 << (PATH_BITS + ENTRY_BITS));

    reg [DATA_BITS-1:0] mem [0:DEPTH-1];
    reg                 valid [0:DEPTH-1];

    wire [PATH_BITS+ENTRY_BITS-1:0] lookup_addr = {lookup_path, lookup_entry};
    wire [PATH_BITS+ENTRY_BITS-1:0] update_addr = {update_path, update_entry};

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lookup_valid <= 1'b0;
            lookup_data <= {DATA_BITS{1'b0}};
            for (i = 0; i < DEPTH; i = i + 1) begin
                valid[i] <= 1'b0;
                mem[i] <= {DATA_BITS{1'b0}};
            end
        end else begin
            if (update_en) begin
                mem[update_addr] <= update_data;
                valid[update_addr] <= 1'b1;
            end
            lookup_valid <= lookup_en && valid[lookup_addr];
            lookup_data <= lookup_en ? mem[lookup_addr] : {DATA_BITS{1'b0}};
        end
    end
endmodule
