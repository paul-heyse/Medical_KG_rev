import http from 'k6/http';
import { Trend, Counter, Rate } from 'k6/metrics';
import { check, sleep } from 'k6';

const baseUrl = __ENV.BASE_URL || 'http://localhost:8000';
const tenantId = __ENV.TENANT_ID || 'load-test';
const apiToken = __ENV.API_TOKEN || '';
const soakDuration = __ENV.SOAK_DURATION || '24h';
const stressArrival = Number(__ENV.STRESS_ARRIVAL || '200');
const soakArrival = Number(__ENV.SOAK_ARRIVAL || '10');
const hybridTopK = Number(__ENV.TOP_K || '10');
const rerankTopK = Number(__ENV.RERANK_TOP_K || '10');

const bm25Latency = new Trend('retrieval_component_bm25', true);
const spladeLatency = new Trend('retrieval_component_splade', true);
const denseLatency = new Trend('retrieval_component_dense', true);
const fusionLatency = new Trend('retrieval_stage_fusion', true);
const rerankLatency = new Trend('retrieval_stage_rerank', true);
const cacheHitRate = new Rate('retrieval_cache_hit_rate');
const failures = new Counter('retrieval_failures_total');

function headers() {
  const values = { 'Content-Type': 'application/json' };
  if (apiToken) {
    values.Authorization = `Bearer ${apiToken}`;
  }
  return values;
}

function performQuery({ rerank = false, queryIntent = null, tableOnly = false }) {
  const payload = {
    tenant_id: tenantId,
    query: queryIntent === 'tabular' ? 'pembrolizumab adverse events' : 'diabetes treatment outcomes',
    top_k: rerank ? rerankTopK : hybridTopK,
    rerank,
    rerank_model: __ENV.RERANK_MODEL || undefined,
    query_intent: queryIntent,
    table_only: tableOnly,
    explain: true,
  };

  const res = http.post(`${baseUrl}/v1/retrieve`, JSON.stringify(payload), { headers: headers() });
  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'has documents': (r) => (r.json('data.documents') || []).length > 0,
  });

  if (!ok) {
    failures.add(1);
    return;
  }

  const meta = res.json('meta') || {};
  const rerankMeta = meta.rerank || {};
  const stageTimings = rerankMeta.stage_timings_ms || {};
  const topLevelStages = meta.stage_timings || {};
  const documents = res.json('data.documents') || [];

  // Component timing extraction from document metadata when available.
  let componentTimings = {};
  for (const doc of documents) {
    const components = doc.metadata && doc.metadata.components;
    if (components && components.timings_ms) {
      componentTimings = components.timings_ms;
      break;
    }
  }

  if (componentTimings.bm25 !== undefined) {
    bm25Latency.add(componentTimings.bm25);
  }
  if (componentTimings.splade !== undefined) {
    spladeLatency.add(componentTimings.splade);
  }
  if (componentTimings.dense !== undefined) {
    denseLatency.add(componentTimings.dense);
  }
  if (topLevelStages.retrieve !== undefined) {
    fusionLatency.add(topLevelStages.retrieve * 1000);
  }
  if (stageTimings.rerank !== undefined) {
    rerankLatency.add(stageTimings.rerank);
  }

  const rerankApplied = rerankMeta.applied === true;
  const cacheInfo = rerankMeta.cache || {};
  if (rerankApplied && cacheInfo.hit !== undefined) {
    cacheHitRate.add(cacheInfo.hit ? 1 : 0);
  }
}

export const options = {
  scenarios: {
    hybrid: {
      executor: 'constant-vus',
      vus: Number(__ENV.HYBRID_VUS || '20'),
      duration: __ENV.HYBRID_DURATION || '3m',
      exec: 'hybridScenario',
      tags: { scenario: 'hybrid' },
    },
    rerank: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RERANK_RATE || '40'),
      timeUnit: '1s',
      duration: __ENV.RERANK_DURATION || '3m',
      preAllocatedVUs: Number(__ENV.RERANK_VUS || '20'),
      maxVUs: Number(__ENV.RERANK_MAX_VUS || '50'),
      exec: 'rerankScenario',
      tags: { scenario: 'rerank' },
    },
    stress: {
      executor: 'ramping-arrival-rate',
      startRate: Number(__ENV.STRESS_START_RATE || '50'),
      timeUnit: '1s',
      stages: [
        { duration: '2m', target: Math.floor(stressArrival / 2) },
        { duration: '2m', target: stressArrival },
        { duration: '2m', target: Math.floor(stressArrival / 4) },
      ],
      preAllocatedVUs: Number(__ENV.STRESS_VUS || '100'),
      maxVUs: Number(__ENV.STRESS_MAX_VUS || '250'),
      exec: 'stressScenario',
      tags: { scenario: 'stress' },
    },
    soak: {
      executor: 'constant-arrival-rate',
      rate: soakArrival,
      timeUnit: '1s',
      duration: soakDuration,
      preAllocatedVUs: Number(__ENV.SOAK_VUS || '25'),
      maxVUs: Number(__ENV.SOAK_MAX_VUS || '50'),
      gracefulStop: '5m',
      exec: 'soakScenario',
      tags: { scenario: 'soak' },
    },
  },
  thresholds: {
    'http_req_duration{scenario:hybrid}': ['p(95)<500'],
    'http_req_duration{scenario:rerank}': ['p(95)<650'],
    'http_req_duration{scenario:stress}': ['p(95)<750'],
    'http_req_duration{scenario:soak}': ['p(95)<600'],
    'retrieval_component_bm25': ['p(95)<100'],
    'retrieval_component_splade': ['p(95)<150'],
    'retrieval_component_dense': ['p(95)<50'],
    'retrieval_stage_fusion': ['p(95)<10'],
    'retrieval_stage_rerank': ['p(95)<150'],
    'retrieval_cache_hit_rate': ['rate>0.4'],
    checks: ['rate>0.98'],
  },
};

export function hybridScenario() {
  performQuery({ rerank: false, queryIntent: null, tableOnly: false });
  sleep(1);
}

export function rerankScenario() {
  performQuery({ rerank: true, queryIntent: null, tableOnly: false });
  sleep(0.5);
}

export function stressScenario() {
  const intents = [null, 'tabular', 'narrative'];
  const intent = intents[Math.floor(Math.random() * intents.length)];
  performQuery({ rerank: Math.random() < 0.3, queryIntent: intent, tableOnly: intent === 'tabular' && Math.random() < 0.2 });
}

export function soakScenario() {
  performQuery({ rerank: true, queryIntent: 'tabular', tableOnly: false });
  sleep(1);
}
