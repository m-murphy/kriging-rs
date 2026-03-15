/**
 * Run in a separate process to assert that using the API before init() throws.
 * Exit code 0 if the expected error is thrown; non-zero otherwise.
 */
import * as kriging from "../dist/index.js";

try {
  kriging.fitOrdinaryVariogram([0], [0], [1], undefined, 1, 2);
  process.exit(1);
} catch (err) {
  const msg = err?.message ?? String(err);
  if (msg.includes("not loaded") && msg.includes("init()")) {
    process.exit(0);
  }
  process.exit(1);
}
