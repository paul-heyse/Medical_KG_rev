import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 1,
  iterations: 5,
  thresholds: {
    http_req_duration: ['p(95)<500'],
  },
};

const BASE_URL = __ENV.GATEWAY_URL || 'http://localhost:8000';

export default function () {
  const ingest = http.post(`${BASE_URL}/v1/ingest/clinicaltrials`, JSON.stringify({
    tenant_id: 'perf',
    items: [{ id: `doc-${__ITER}` }],
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(ingest, { 'ingest status is 207': (r) => r.status === 207 });

  const retrieve = http.post(`${BASE_URL}/v1/retrieve`, JSON.stringify({
    tenant_id: 'perf',
    query: 'cancer',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(retrieve, { 'retrieve status is 200': (r) => r.status === 200 });

  sleep(0.5);
}
