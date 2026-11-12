# Framework figure

The homepage embeds the manuscript's symbol-corrected framework exported on November 12, 2026. The PNG, SVG, and PDF are copied without graphical modifications from that export.

The diagram distinguishes input features X = H^(0), post-ReLU/post-dropout H^(1), normalized graph operators, layer weights, and detached graph construction. The first-layer operator is cached; the second is rebuilt from the current hidden representation. Local graph contributions are summed, I_n is added once, and symmetric normalization follows. Logits U feed argmax prediction.

This figure documents architecture and graph construction; it is not an experimental result or evidence of bitwise numerical reproducibility. See REPRODUCIBILITY.md for validation scope.
