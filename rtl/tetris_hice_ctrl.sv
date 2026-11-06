`timescale 1ns/1ps

module tetris_hice_ctrl (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       begin_update,
    input  wire       commit_req,
    input  wire       rollback_req,
    input  wire       queue_empty,
    output reg        queue_pop,
    output reg        commit_pulse,
    output reg        rollback_pulse,
    output reg        busy,
    output reg [1:0]  state_dbg
);
    localparam S_IDLE      = 2'd0;
    localparam S_TENTATIVE = 2'd1;
    localparam S_COMMIT    = 2'd2;
    localparam S_ROLLBACK  = 2'd3;

    reg [1:0] state;
    reg [1:0] next_state;

    always @(*) begin
        next_state = state;
        queue_pop = 1'b0;
        commit_pulse = 1'b0;
        rollback_pulse = 1'b0;
        busy = (state != S_IDLE);
        case (state)
            S_IDLE: begin
                if (begin_update) next_state = S_TENTATIVE;
            end
            S_TENTATIVE: begin
                if (commit_req) next_state = S_COMMIT;
                else if (rollback_req) next_state = S_ROLLBACK;
            end
            S_COMMIT: begin
                commit_pulse = 1'b1;
                queue_pop = !queue_empty;
                next_state = S_IDLE;
            end
            S_ROLLBACK: begin
                rollback_pulse = 1'b1;
                queue_pop = !queue_empty;
                next_state = S_IDLE;
            end
            default: next_state = S_IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            state_dbg <= S_IDLE;
        end else begin
            state <= next_state;
            state_dbg <= next_state;
        end
    end
endmodule
