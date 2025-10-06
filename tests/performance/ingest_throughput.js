import http from 'k6/http';
import { check } from 'k6';

export const options = {
  scenarios: {
    ingest: {
      executor: 'constant-arrival-rate',
      rate: 10,
      timeUnit: '1s',
      duration: '1m',
      preAllocatedVUs: 5,
      maxVUs: 20,
    },
  },
  thresholds: {
    'http_req_duration{scenario:ingest}': ['p(95)<800'],
    'checks{scenario:ingest}': ['rate>0.9'],
  },
};

const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const payload = JSON.stringify({
    tenant_id: 'load-test',
    items: [
      { id: `doc-${__ITER}`, title: 'Synthetic Study', url: 'http://example.com/doc' },
      { id: `doc-${__ITER}-b`, title: 'Synthetic Study B', url: 'http://example.com/doc-b' },
    ],
  });
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(`${baseUrl}/v1/ingest/clinicaltrials`, payload, params);
  check(res, {
    'status is 207': (r) => r.status === 207,
  });
}
