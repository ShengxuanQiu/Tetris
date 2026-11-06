`timescale 1ns/1ps

module tetris_desc_queue #(
    parameter DEPTH = 8,
    parameter DATA_BITS = 64,
    parameter PTR_BITS = 4
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 push,
    input  wire [DATA_BITS-1:0] push_data,
    input  wire                 pop,
    output reg  [DATA_BITS-1:0] pop_data,
    output wire                 empty,
    output wire                 full,
    output reg  [PTR_BITS:0]    count
);
    reg [DATA_BITS-1:0] fifo [0:DEPTH-1];
    reg [PTR_BITS-1:0] head;
    reg [PTR_BITS-1:0] tail;

    assign empty = (count == 0);
    assign full = (count == DEPTH);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= {PTR_BITS{1'b0}};
            tail <= {PTR_BITS{1'b0}};
            count <= {(PTR_BITS+1){1'b0}};
            pop_data <= {DATA_BITS{1'b0}};
        end else begin
            if (push && !full) begin
                fifo[tail] <= push_data;
                if (tail == DEPTH-1)
                    tail <= {PTR_BITS{1'b0}};
                else
                    tail <= tail + 1'b1;
            end
            if (pop && !empty) begin
                pop_data <= fifo[head];
                if (head == DEPTH-1)
                    head <= {PTR_BITS{1'b0}};
                else
                    head <= head + 1'b1;
            end
            case ({push && !full, pop && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
