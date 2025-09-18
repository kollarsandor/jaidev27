module neuromorphic_pulse_array (
    input clk,
    input reset
);
    wire [15:0] tgn_reg;
    wire [15:0] tgn_control;

    J1Core j1 (
        .clk(clk),
        .reset(reset),
        .tgn_reg(tgn_reg),
        .tgn_control(tgn_control),
        .mem_addr(),
        .mem_data_in(16'b0),
        .mem_data_out(),
        .mem_write()
    );
endmodule
