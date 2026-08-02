import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';

// ---- Config ----
const SERVICES = {
  fast: 'http://localhost:8001/predict',
  balanced: 'http://localhost:8002/predict',
  heavy: 'http://localhost:8003/predict',
};

// Which tier to hit -- set via k6 env var: k6 run -e TIER=fast load_test.js
const TIER = __ENV.TIER || 'fast';
const TARGET_URL = SERVICES[TIER];

// Test image -- must exist on disk, k6 reads it once and reuses the bytes
const IMAGE_PATH = __ENV.IMAGE_PATH || './sample_images/test_apple.png';

const imageData = open(IMAGE_PATH, 'b');

// ---- Load stages: light -> heavy -> burst ----
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
  const formData = {
    file: http.file(imageData, 'test.png', 'image/png'),
  };

  const res = http.post(TARGET_URL, formData);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has predicted_class': (r) => {
      try {
        return JSON.parse(r.body).predicted_class !== undefined;
      } catch {
        return false;
      }
    },
  });

  sleep(0.1); // small pacing so we don't fully saturate instantly
}