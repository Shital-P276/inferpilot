import http from 'k6/http';
import { check, sleep } from 'k6';

// ---- Config ----
const GATEWAY_URL = 'http://localhost:8000/route';

// Strategy selected via: k6 run -e STRATEGY=always_fast load_test_strategies.js
const STRATEGY = __ENV.STRATEGY || 'ml_router';

const IMAGE_PATH = __ENV.IMAGE_PATH || './sample_images/test_apple.png';
const imageData = open(IMAGE_PATH, 'b');

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
  const formData = {
    file: http.file(imageData, 'test.png', 'image/png'),
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