`timescale 1ns/1ps

module tb_tetris_controller_sanity;
    reg clk;
    reg rst_n;
    reg lookup_en;
    reg [7:0] lookup_path;
    reg [3:0] lookup_entry;
    wire lookup_valid;
    wire [63:0] lookup_data;
    reg update_en;
    reg [7:0] update_path;
    reg [3:0] update_entry;
    reg [63:0] update_data;
    reg desc_push;
    reg [63:0] desc_data;
    reg begin_update;
    reg commit_req;
    reg rollback_req;
    wire commit_pulse;
    wire rollback_pulse;
    wire busy;
    wire [1:0] hice_state_dbg;
    wire [4:0] desc_count;
    wire [63:0] desc_pop_data;
    integer commit_seen;
    integer rollback_seen;

    tetris_controller_sanity_top dut (
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

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n) begin
            commit_seen <= 0;
            rollback_seen <= 0;
        end else begin
            if (commit_pulse) commit_seen <= commit_seen + 1;
            if (rollback_pulse) rollback_seen <= rollback_seen + 1;
        end
    end

    task fail;
        input [255:0] msg;
        begin
            $display("FAIL: %0s", msg);
            $finish;
        end
    endtask

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        lookup_en = 1'b0;
        lookup_path = 8'd0;
        lookup_entry = 4'd0;
        update_en = 1'b0;
        update_path = 8'd0;
        update_entry = 4'd0;
        update_data = 64'd0;
        desc_push = 1'b0;
        desc_data = 64'd0;
        begin_update = 1'b0;
        commit_req = 1'b0;
        rollback_req = 1'b0;
        commit_seen = 0;
        rollback_seen = 0;
        $dumpfile("tetris_controller_sanity.vcd");
        $dumpvars(0, tb_tetris_controller_sanity);
        repeat (3) tick();
        rst_n = 1'b1;
        tick();

        update_path = 8'h12;
        update_entry = 4'h3;
        update_data = 64'hDEAD_BEEF_0000_0001;
        update_en = 1'b1;
        tick();
        update_en = 1'b0;
        lookup_path = 8'h12;
        lookup_entry = 4'h3;
        lookup_en = 1'b1;
        tick();
        if (!lookup_valid || lookup_data != 64'hDEAD_BEEF_0000_0001) begin
            fail("HATT lookup/update mismatch");
        end
        lookup_en = 1'b0;

        desc_data = 64'hCAFE_0000_0000_0001;
        desc_push = 1'b1;
        tick();
        desc_push = 1'b0;
        if (desc_count != 5'd1) begin
            fail("descriptor queue push count mismatch");
        end

        begin_update = 1'b1;
        tick();
        begin_update = 1'b0;
        if (!busy) begin
            fail("HICE did not enter tentative state");
        end

        commit_req = 1'b1;
        tick();
        commit_req = 1'b0;
        tick();
        if (commit_seen != 1 || desc_count != 5'd0 || desc_pop_data != 64'hCAFE_0000_0000_0001) begin
            fail("HICE commit did not emit one pulse and drain descriptor");
        end

        desc_data = 64'hCAFE_0000_0000_0002;
        desc_push = 1'b1;
        tick();
        desc_push = 1'b0;
        begin_update = 1'b1;
        tick();
        begin_update = 1'b0;
        rollback_req = 1'b1;
        tick();
        rollback_req = 1'b0;
        tick();
        if (rollback_seen != 1 || desc_count != 5'd0 || desc_pop_data != 64'hCAFE_0000_0000_0002) begin
            fail("HICE rollback did not emit one pulse and drain descriptor");
        end

        $display("PASS: minimal Tetris controller-side RTL sanity test");
        $finish;
    end
endmodule
