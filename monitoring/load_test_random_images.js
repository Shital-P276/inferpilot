/*
load_test_random_images.js

Same load-stage pattern as load_test_strategies.js (light -> heavy -> burst),
but sends VARIED real images instead of repeating one file over and over.
This exercises the router's actual per-image decision-making under load,
not just raw throughput -- you should see Predictions by Class and the
Predictions by Tier pie chart reflect real content diversity, not a single
spike.

HOW THIS WORKS AROUND K6's CONSTRAINT:
k6 can only read files during INIT (before the test starts), not inside
the default function during the actual run. So this script:
  1. Reads router/held_out_images.csv (relative to project root) ONCE at
     init time, using open() -- this itself only works because .csv is a
     text file k6 can open like any other.
  2. Parses out the image_path column, takes a random sample of N images.
  3. Opens each of those N image files (also at init time) into an array.
  4. During the actual test, each virtual user randomly picks one already-
     loaded image per request -- no filesystem access happens mid-test.

BEFORE RUNNING:
    Confirm router/held_out_images.csv exists relative to project root
    (it should, from earlier work in this project) and that its
    image_path column values are real, valid paths under data/test/.

USAGE:
    k6 run monitoring/load_test_random_images.js
    k6 run -e STRATEGY=always_heavy monitoring/load_test_random_images.js
    k6 run -e N_IMAGES=50 monitoring/load_test_random_images.js
*/

import http from 'k6/http';
import { check, sleep } from 'k6';

// ---- Config ----
const GATEWAY_URL = 'http://localhost:8000/route';
const STRATEGY = __ENV.STRATEGY || 'ml_router';
const N_IMAGES = parseInt(__ENV.N_IMAGES || '250', 10);

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

// ---- Load stages: same light -> heavy -> burst pattern as prior tests ----
export const options = {
  scenarios: {
    light_load: {
      executor: 'constant-vus',
      vus: 5,
      duration: '30s',
      startTime: '0s',
      tags: { load_stage: 'light' },
    },
    heavy_load: {
      executor: 'constant-vus',
      vus: 25,
      duration: '30s',
      startTime: '35s',
      tags: { load_stage: 'heavy' },
    },
    burst_load: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '10s', target: 50 },
        { duration: '15s', target: 50 },
        { duration: '5s', target: 0 },
      ],
      startTime: '70s',
      tags: { load_stage: 'burst' },
    },
  },
};

export default function () {
  const chosen = loadedImages[Math.floor(Math.random() * loadedImages.length)];

  const formData = {
    file: http.file(chosen.data, 'test.jpg', 'image/jpeg'),
  };

  const url = `${GATEWAY_URL}?strategy=${STRATEGY}`;
  const res = http.post(url, formData, { tags: { strategy: STRATEGY } });

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