import { useState } from "react";
import {
  OrdinaryKriging,
  BinomialKriging,
  VariogramType,
  fitOrdinaryVariogram,
} from "kriging-rs-wasm";
import { buildBinomialLogits } from "../lib/sampleData";

export default function QuickDemos({ onError }) {
  const [output, setOutput] = useState(null);
  const [running, setRunning] = useState(false);

  function write(title, payload) {
    setOutput({ title, payload });
  }

  function runOrdinary() {
    setRunning(true);
    try {
      const lats = [37.77, 37.78, 37.76, 37.75];
      const lons = [-122.42, -122.41, -122.4, -122.43];
      const values = [15, 18, 14, 13];
      const model = new OrdinaryKriging(lats, lons, values, "exponential", 0.1, 6.0, 5.0);
      const pred = model.predict(37.765, -122.415);
      model.free();
      write("Ordinary kriging prediction", pred);
    } catch (err) {
      onError?.(err?.message ?? String(err));
      write("Ordinary kriging failed", { message: err?.message ?? String(err) });
    } finally {
      setRunning(false);
    }
  }

  function runBinomial() {
    setRunning(true);
    try {
      const lats = [40.71, 40.72, 40.7];
      const lons = [-74.0, -73.99, -74.02];
      const successes = [25, 35, 20];
      const trials = [50, 50, 50];
      const model = new BinomialKriging(
        lats,
        lons,
        successes,
        trials,
        "gaussian",
        0.01,
        1.5,
        10.0,
      );
      const pred = model.predict(40.715, -74.005);
      model.free();
      write("Binomial kriging prediction", pred);
    } catch (err) {
      onError?.(err?.message ?? String(err));
      write("Binomial kriging failed", { message: err?.message ?? String(err) });
    } finally {
      setRunning(false);
    }
  }

  function runFittedPipeline() {
    setRunning(true);
    try {
      const nBins = 12;
      const maxDistance = undefined;
      const variogramType = VariogramType.Exponential;
      const alpha = 0.5;
      const beta = 0.5;

      const sampleLats = [0.0, 0.5, 1.0];
      const sampleLons = [0.0, 0.5, 1.0];
      const values = [2.0, 3.0, 4.0];
      const predLats = [0.75, 0.25];
      const predLons = [0.75, 0.25];

      const ordinaryFit = fitOrdinaryVariogram(
        sampleLats,
        sampleLons,
        values,
        maxDistance,
        nBins,
        variogramType,
      );
      const ordinaryModel = new OrdinaryKriging(
        sampleLats,
        sampleLons,
        values,
        ordinaryFit.variogramType,
        ordinaryFit.nugget,
        ordinaryFit.sill,
        ordinaryFit.range,
        ordinaryFit.shape,
      );
      const ordinaryOut = ordinaryModel.predictBatch(predLats, predLons);
      ordinaryModel.free();

      const sampleSuccesses = [3, 6, 8];
      const sampleTrials = [10, 10, 10];
      const sampleLogits = buildBinomialLogits(
        sampleSuccesses,
        sampleTrials,
        alpha,
        beta,
      );
      const binomialFit = fitOrdinaryVariogram(
        sampleLats,
        sampleLons,
        sampleLogits,
        maxDistance,
        nBins,
        variogramType,
      );
      const binomialModel = BinomialKriging.newWithPrior(
        sampleLats,
        sampleLons,
        sampleSuccesses,
        sampleTrials,
        binomialFit.variogramType,
        binomialFit.nugget,
        binomialFit.sill,
        binomialFit.range,
        alpha,
        beta,
        binomialFit.shape,
      );
      const binomialOut = binomialModel.predictBatch(predLats, predLons);
      binomialModel.free();
      write("Fitted pipeline output", {
        ordinaryOut,
        binomialOut,
        options: { nBins, alpha, beta },
      });
    } catch (err) {
      onError?.(err?.message ?? String(err));
      write("Fitted pipeline demo failed", { message: err?.message ?? String(err) });
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="panel">
      <h2>Quick demos</h2>
      <p>
        Run a single ordinary or binomial prediction, or the full fitted pipeline (fit
        variogram then predict).
      </p>
      <div>
        <button type="button" onClick={runOrdinary} disabled={running}>
          Ordinary kriging
        </button>
        <button type="button" onClick={runBinomial} disabled={running}>
          Binomial kriging
        </button>
        <button type="button" onClick={runFittedPipeline} disabled={running}>
          Fitted pipeline
        </button>
      </div>
      <div style={{ marginTop: "1rem" }}>
        <strong>Output</strong>
        <pre>{output ? `${output.title}\n\n${JSON.stringify(output.payload, null, 2)}` : "Click a button to run."}</pre>
      </div>
    </div>
  );
}
