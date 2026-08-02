# Configuration reference

DualArrow uses a validated `dualarrow.yaml` file. Unknown fields are rejected.

## Environment variables

Use `${NAME}` for required values and `${NAME:-default}` for optional defaults.

```yaml
security:
  api_key: ${DUALARROW_API_KEY}
```

## Main sections

- `project`
- `knowledge_store`
- `stores`
- `ingestion`
- `embedding`
- `generation`
- `retrieval_profiles`
- `default_profile`
- `verification`
- `security`
- `api`

Run `dualarrow config explain dualarrow.yaml` to display the resolved configuration with defaults.
