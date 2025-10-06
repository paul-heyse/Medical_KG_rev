import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 5,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(95)<500'],
    checks: ['rate>0.95'],
  },
};

const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const payload = JSON.stringify({ tenant_id: 'load-test', query: 'oncology', top_k: 3 });
  const params = { headers: { 'Content-Type': 'application/json' } };
  const res = http.post(`${baseUrl}/v1/retrieve`, payload, params);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'returned documents': (r) => (r.json('data') || []).length >= 1,
  });
  sleep(1);
}
