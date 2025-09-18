module J1Core (
    input clk,
    input reset,
    output reg [15:0] mem_addr,
    input [15:0] mem_data_in,
    output reg [15:0] mem_data_out,
    output reg mem_write,
    input [15:0] tgn_reg,
    output reg [15:0] tgn_control
);
    reg [15:0] pc, sp, rp;
    reg [15:0] stack [0:31];
    reg [15:0] rstack [0:31];
    reg [15:0] insn;
    reg [3:0] dsp, rsp;
    reg [15:0] st0;
    reg [15:0] rst0;
    reg reboot;

    initial begin
        reboot = 1;
    end

    always @(posedge clk) begin
        if (reset || reboot) begin
            pc <= 0;
            sp <= 0;
            rp <= 0;
            dsp <= 0;
            rsp <= 0;
            reboot <= 0;
            mem_write <= 0;
            mem_addr <= 0;
            mem_data_out <= 0;
            tgn_control <= 0;
        end else begin
            mem_addr <= pc;
            insn <= mem_data_in;
            casez (insn[15:13])
                3'b0??: begin // JMP
                    pc <= insn[12:0];
                end
                3'b1??: begin // CALL
                    rstack[rsp] <= pc + 1;
                    rsp <= rsp + 1;
                    pc <= insn[12:0];
                end
                3'b2??: begin // JZ
                    if (st0 == 0)
                        pc <= insn[12:0];
                    else
                        pc <= pc + 1;
                    st0 <= stack[dsp - 1];
                    dsp <= dsp - 1;
                end
                3'b3??: begin // ALU
                    case (insn[7:0])
                        8'h00: st0 <= st0; // T
                        8'h01: st0 <= stack[dsp - 1]; // N
                        8'h02: st0 <= st0 + stack[dsp - 1]; // T+N
                        8'h03: st0 <= st0 & stack[dsp - 1]; // T&N
                        default: st0 <= st0;
                    endcase
                    pc <= pc + 1 + insn[12];
                    if (insn[11]) begin
                        mem_data_out <= st0;
                        mem_write <= 1;
                    end else begin
                        mem_write <= 0;
                    end
                end
                default: pc <= pc + 1;
            endcase
            tgn_control <= tgn_reg;
        end
    end
endmodule
