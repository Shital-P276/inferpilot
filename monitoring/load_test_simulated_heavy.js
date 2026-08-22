/*
load_test_simulated_heavy.js

Widened-gap stress test: same light -> heavy -> burst pattern and same
CSV-sampling-at-init-time image loading as load_test_random_images.js, but
runs against FOUR strategies in one execution -- always_heavy, ml_router,
rule_based, correctness_gate_router -- the strategies whose %-routed-to-Heavy
differs meaningfully (per router/core_experiment_summary.csv). The point is to
see whether the routers earn a measurable latency advantage once Heavy's
latency resembles a real cloud vision API.

!!! CRITICAL PREREQUISITE !!!
This test MUST be run against a gateway container started with the simulated
Heavy delay env var set:
    SIMULATED_HEAVY_DELAY_MS=2500
(exact env var name confirmed against serving/gateway_service.py). Running it
against a gateway WITHOUT that var set produces baseline (non-simulated,
~6.29ms-Heavy) numbers, silently making any comparison meaningless. There is no
way for k6 to confirm the gateway's env var from the client side -- verify it
before trusting any result. (A console.warn is emitted at init as a reminder.)

STRUCTURE -- sequential strategy cycles, not concurrent:
Each strategy runs its own full light(5 VUs, 30s) -> heavy(25 VUs, 30s) ->
burst(ramping to 50 VUs, 30s) cycle (~100s). The four cycles are staggered via
scenario startTime so only ONE strategy is hitting the gateway at a time --
results are not confounded by two strategies running concurrently. Every
request is tagged with both strategy and load_stage so Prometheus data can be
filtered by either dimension afterward.

USAGE:
    k6 run monitoring/load_test_simulated_heavy.js
    k6 run -e N_IMAGES=50 monitoring/load_test_simulated_heavy.js
*/

import http from 'k6/http';
import { check, sleep } from 'k6';
import exec from 'k6/execution';

// ---- Config ----
const GATEWAY_URL = 'http://localhost:8000/route';
const N_IMAGES = parseInt(__ENV.N_IMAGES || '250', 10);

// Strategies whose %-routed-to-Heavy differs meaningfully per
// router/core_experiment_summary.csv. One full light->heavy->burst cycle per
// strategy, sequential.
const STRATEGIES = ['always_heavy', 'ml_router', 'rule_based', 'correctness_gate_router'];

function strategyFromScenarioName(scenarioName) {
  for (const strategy of STRATEGIES) {
    if (scenarioName === `${strategy}_light` || scenarioName === `${strategy}_heavy` || scenarioName === `${strategy}_burst`) {
      return strategy;
    }
  }
  throw new Error(`Could not derive strategy from scenario name "${scenarioName}" -- expected one of STRATEGIES with a _light/_heavy/_burst suffix.`);
}

// held_out_images.csv is at the project root's router/ folder. This script
// lives in monitoring/, so paths are relative to THIS script's location --
// k6's open() resolves relative paths relative to the script file, not the
// current working directory.
const CSV_PATH = '../router/held_out_images.csv';

// ---- INIT: parse the CSV and pre-load a random sample of real images ----
function parseCsv(text) {
  const lines = text.trim().split('\n');
  const header = lines[0].split(',');
  const pathIdx = header.indexOf('image_path');
  if (pathIdx === -1) {
    throw new Error(
      `Expected an 'image_path' column in ${CSV_PATH}, got header: ${header.join(',')}`
    );
  }
  return lines.slice(1).map((line) => line.split(',')[pathIdx]).filter(Boolean);
}

function shuffleAndTake(arr, n) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

const csvText = open(CSV_PATH);
const allPaths = parseCsv(csvText);
const sampledPaths = shuffleAndTake(allPaths, Math.min(N_IMAGES, allPaths.length));

// Load each sampled image's binary data at INIT time. Paths in the CSV use
// Windows-style backslashes and are relative to the PROJECT ROOT (e.g.
// "data\test\Apple__Healthy\foo.jpg") -- convert to forward slashes and
// prefix with "../" since this script sits one level down in monitoring/.
const loadedImages = [];
for (const rawPath of sampledPaths) {
  const relativePath = '../' + rawPath.replace(/\\/g, '/');
  try {
    const data = open(relativePath, 'b');
    loadedImages.push({ path: rawPath, data });
  } catch (e) {
    // Skip images that fail to load (e.g. path mismatch) rather than
    // crashing the whole init -- print so it's visible, not silent.
    console.warn(`Could not open ${relativePath}: ${e}`);
  }
}

if (loadedImages.length === 0) {
  throw new Error(
    `No images successfully loaded from ${CSV_PATH}. Check CSV_PATH and that ` +
    `the image_path values actually resolve relative to this script's location.`
  );
}

console.log(`Loaded ${loadedImages.length}/${sampledPaths.length} sampled images for the load test.`);
console.warn(
  `REMINDER: this test measures the WIDENED Heavy-latency gap. Confirm the ` +
  `gateway container was started with SIMULATED_HEAVY_DELAY_MS=2500 set. If it ` +
  `was NOT, this run silently produces baseline (non-simulated) numbers and any ` +
  `comparison against the real baseline is meaningless.`
);

// ---- Load stages: same light -> heavy -> burst pattern as prior tests ----
// One cycle per strategy (~100s), staggered so strategy cycles never overlap.
const CYCLE_START_S = 100;
const CYCLE_SECONDS = {
  light: 0,
  heavy: 35,
  burst: 70,
};

const scenarios = {};
for (let s = 0; s < STRATEGIES.length; s++) {
  const strategy = STRATEGIES[s];
  const base = s * CYCLE_START_S;

  scenarios[`${strategy}_light`] = {
    executor: 'constant-vus',
    vus: 5,
    duration: '30s',
    startTime: `${base + CYCLE_SECONDS.light}s`,
    tags: { load_stage: 'light', strategy },
  };

  scenarios[`${strategy}_heavy`] = {
    executor: 'constant-vus',
    vus: 25,
    duration: '30s',
    startTime: `${base + CYCLE_SECONDS.heavy}s`,
    tags: { load_stage: 'heavy', strategy },
  };

  scenarios[`${strategy}_burst`] = {
    executor: 'ramping-vus',
    startVUs: 5,
    stages: [
      { duration: '10s', target: 50 },
      { duration: '15s', target: 50 },
      { duration: '5s', target: 0 },
    ],
    startTime: `${base + CYCLE_SECONDS.burst}s`,
    tags: { load_stage: 'burst', strategy },
  };
}

export const options = { scenarios };

export default function () {
  // exec.scenario.tags is not part of k6's runtime API (scenario-level tags in
  // options.scenarios are metric metadata, not readable back at runtime) -- so
  // derive the strategy from the scenario name, which encodes it as a prefix.
  const strategy = strategyFromScenarioName(exec.scenario.name);

  const chosen = loadedImages[Math.floor(Math.random() * loadedImages.length)];

  const formData = {
    file: http.file(chosen.data, 'test.jpg', 'image/jpeg'),
  };

  const url = `${GATEWAY_URL}?strategy=${strategy}`;
  const res = http.post(url, formData, { tags: { strategy } });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has routed_tier': (r) => {
      try {
        return JSON.parse(r.body).routed_tier !== undefined;
      } catch {
        return false;
      }
    },
  });

  sleep(0.1);
}