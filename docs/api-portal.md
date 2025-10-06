# Medical KG API Portal

This lightweight developer portal provides quick access to the REST, GraphQL, gRPC, SOAP, and AsyncAPI documentation for the multi-protocol gateway.

## REST (OpenAPI)

- Interactive Swagger UI: `/docs/openapi`
- Specification file: [`docs/openapi.yaml`](./openapi.yaml)
- Contract testing recommendation: `pytest tests/contract/test_rest_contract.py`

## GraphQL

- Playground: `/docs/graphql`
- Endpoint: `/graphql`
- Schema SDL: [`docs/schema.graphql`](./schema.graphql)
- Contract testing recommendation: `pytest tests/contract/test_graphql_contract.py`

## gRPC

- Proto definitions under [`src/Medical_KG_rev/proto`](../src/Medical_KG_rev/proto)
- Buf configuration: [`buf.yaml`](../buf.yaml)
- Code generation: `buf generate`
- Health check service exposed via `grpc.health.v1`

## AsyncAPI (SSE)

- AsyncAPI UI: `/docs/asyncapi`
- Specification: [`docs/asyncapi.yaml`](./asyncapi.yaml)
- Authentication: provide `X-API-Key` header when connecting to `/v1/jobs/{job_id}/events`

## SOAP

- SOAP endpoint: `/soap`
- WSDL document: `/soap/wsdl`

## Authentication Guide

All protocols expect a tenant-aware request. Include the `tenant_id` field in request bodies. For SSE streaming, add the `X-API-Key` header with `public-demo-key`.

## Example Workflow

1. **Ingest:** Submit documents via REST `/v1/ingest/clinicaltrials`
2. **Chunk:** Call GraphQL `chunk` mutation to segment documents
3. **Embed:** Generate embeddings via gRPC `EmbeddingService`
4. **Extract:** Use REST `/v1/extract/pico` to obtain structured claims
5. **Write:** Persist relationships via GraphQL `write_kg` mutation
6. **Stream:** Monitor job status with SSE `/v1/jobs/{job_id}/events`

Refer to the `tests/contract` folder for executable examples.

## Deployment

Serve the API portal and generated specifications locally with:

```bash
python -m http.server --directory docs 9000
```

For GitHub Pages, configure the repository to publish the `docs/` directory as a static site.
