# Gateway Performance Smoke Test

Run the k6 script to exercise core endpoints:

```bash
k6 run tests/performance/gateway_smoke_test.js
```

Set `GATEWAY_URL` to target a non-default gateway instance.
