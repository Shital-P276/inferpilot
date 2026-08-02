import http from 'k6/http';
import { check, sleep } from 'k6';

// ---- Config ----
const SERVICES = {
  fast: 'http://localhost:8001/predict',
  balanced: 'http://localhost:8002/predict',
  heavy: 'http://localhost:8003/predict',
};

const IMAGE_PATH = __ENV.IMAGE_PATH || './sample_images/test_apple.png';
const imageData = open(IMAGE_PATH, 'b');

// ---- Load stages: same light -> heavy -> burst pattern ----
export const options = {
  scenarios: {
    light_load: {
      executor: 'constant-vus',
      vus: 15,          // split roughly 5 per tier
      duration: '30s',
      startTime: '0s',
      tags: { load_stage: 'light' },
    },
    heavy_load: {
      executor: 'constant-vus',
      vus: 75,           // ~25 per tier
      duration: '30s',
      startTime: '35s',
      tags: { load_stage: 'heavy' },
    },
    burst_load: {
      executor: 'ramping-vus',
      startVUs: 15,
      stages: [
        { duration: '10s', target: 150 },  // ~50 per tier
        { duration: '15s', target: 150 },
        { duration: '5s', target: 0 },
      ],
      startTime: '70s',
      tags: { load_stage: 'burst' },
    },
  },
};

// Each VU picks a tier round-robin based on its own VU id,
// so load spreads evenly across all 3 services within each scenario
const TIER_NAMES = Object.keys(SERVICES);

export default function () {
  const tier = TIER_NAMES[__VU % TIER_NAMES.length];
  const url = SERVICES[tier];

  const formData = {
    file: http.file(imageData, 'test.png', 'image/png'),
  };

  const res = http.post(url, formData, { tags: { tier: tier } });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has predicted_class': (r) => {
      try {
        return JSON.parse(r.body).predicted_class !== undefined;
      } catch {
        return false;
      }
    },
  }, { tier: tier });

  sleep(0.1);
}