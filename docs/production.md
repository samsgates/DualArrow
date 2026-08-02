# Production deployment checklist

- Replace the hash embedding provider with a trained embedding model.
- Configure an evidence-constrained generation provider or use extractive mode.
- Pin container images and Python dependencies.
- Enable TLS for every database and API connection.
- Use managed secrets.
- Restrict source and database credentials to read-only permissions where possible.
- Configure database backups.
- Monitor ingestion failures and adapter health.
- Create representative evaluation and adversarial datasets.
- Establish confidence and abstention thresholds per domain.
- Require expert review for medical, legal, financial, or safety-critical answers.
- Add an outbox or queue when synchronizing several external indexes at scale.
