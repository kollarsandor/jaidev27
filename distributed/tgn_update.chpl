module TGNUpdate {
    config const size = 2048;

    proc tgn_update(nodes: [0..size-1] real, memory: [0..size-1] real, time_factor: real): [0..size-1] real {
        var updated: [0..size-1] real;
        forall i in 0..size-1 do
            updated[i] = memory[i] * (1.0 - time_factor) + nodes[i] * time_factor;
        return updated;
    }

    proc main() {
        var nodes: [0..size-1] real = [i: real in 0..size-1] i / size;
        var memory: [0..size-1] real = [i: real in 0..size-1] i / (2 * size);
        var time_factor: real = 0.5;
        var result = tgn_update(nodes, memory, time_factor);
        writeln("TGN update completed.");
    }
}
