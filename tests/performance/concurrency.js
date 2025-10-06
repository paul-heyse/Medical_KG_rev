import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    retrieve: {
      executor: 'ramping-vus',
      stages: [
        { duration: '30s', target: 5 },
        { duration: '30s', target: 10 },
        { duration: '30s', target: 0 },
      ],
    },
    ingest: {
      executor: 'ramping-arrival-rate',
      timeUnit: '1s',
      preAllocatedVUs: 5,
      maxVUs: 20,
      stages: [
        { duration: '30s', target: 5 },
        { duration: '30s', target: 10 },
        { duration: '30s', target: 0 },
      ],
    },
  },
  thresholds: {
    'http_req_duration{scenario:retrieve}': ['p(95)<500'],
    'http_req_duration{scenario:ingest}': ['p(95)<900'],
    'checks{scenario:retrieve}': ['rate>0.9'],
    'checks{scenario:ingest}': ['rate>0.85'],
  },
};

const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';

export function retrieve() {
  const payload = JSON.stringify({ tenant_id: 'load-test', query: 'precision medicine', top_k: 2 });
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(`${baseUrl}/v1/retrieve`, payload, params);
  check(res, {
    'retrieve status 200': (r) => r.status === 200,
  });
  sleep(0.5);
}

export function ingest() {
  const payload = JSON.stringify({
    tenant_id: 'load-test',
    items: [
      { id: `doc-${__ITER}`, title: 'Concurrent A', url: 'http://example.com/a' },
    ],
  });
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(`${baseUrl}/v1/ingest/clinicaltrials`, payload, params);
  check(res, {
    'ingest status 207': (r) => r.status === 207,
  });
}
