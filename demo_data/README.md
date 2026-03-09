# Demo Data

This directory contains **sample data fixtures** for demonstrating and testing the Episodic Memory Pipeline.

## ⚠️ Data Policy

### What belongs here

- **Synthetic data only**: All content must be fictional and generated for demonstration purposes
- **Example memories**: Sample diary entries, learning logs, work notes (all fake)
- **Test fixtures**: JSON files with predictable content for reproducible testing

### What does NOT belong here

- **Real user data**: Never commit actual personal memories or conversations
- **Sensitive information**: No API keys, passwords, PII, or confidential content
- **Production exports**: Never dump real database contents here

## Files

### `fixtures.json`

Contains sample episodes for bootstrapping demos and running evaluations.

Structure:

```json
{
  "version": "1.0",
  "description": "...",
  "episodes": [
    {
      "text": "...",
      "source": "demo",
      "expected_topics": ["topic1", "topic2"]
    }
  ]
}
```

### `eval_queries.json`

Contains evaluation queries with expected results for reproducible testing.

## Regenerating Indexes

After modifying fixtures, regenerate the demo data:

```bash
# From project root
python -m scripts.bootstrap_demo --clean

# Or using make
make demo-clean
make demo
```

This will:

1. Clear existing data in `data/`
2. Re-ingest all fixtures
3. Rebuild LanceDB tables/indexes
4. Print summary of generated artifacts

## Adding New Fixtures

When adding new demo data:

1. Ensure all content is **completely synthetic**
2. Use plausible but **fictional** names, dates, and places
3. Add corresponding queries to `eval_queries.json` if needed
4. Run `make demo` to verify ingestion works
5. Update this README if adding new file types

## Testing Redaction

If testing redaction functionality:

1. Place test content in fixtures with intentional "sensitive-looking" patterns
2. Verify redaction catches them without breaking output format
3. See `tests/test_redaction.py` for examples
