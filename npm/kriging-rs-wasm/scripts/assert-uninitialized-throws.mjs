/**
 * Run in a separate process to assert that using the API before init() throws.
 * Exit code 0 if the expected error is thrown; non-zero otherwise.
 */
import * as kriging from "../dist/index.js";

try {
  // variogramType: 2 is an arbitrary enum value; only used to trigger the "module not loaded" path
  kriging.fitVariogram({ sampleLats: [0], sampleLons: [0], values: [1], variogramType: 2 });
  process.exit(1);
} catch (err) {
  const msg = err?.message ?? String(err);
  if (msg.includes("not loaded") && msg.includes("init()")) {
    process.exit(0);
  }
  process.exit(1);
}
