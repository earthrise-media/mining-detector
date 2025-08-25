To make requests for the Mining Calculator API, you'll need to copy the .env file in the root of this repository and add in your API key:

```bash
MINING_CALCULATOR_API_KEY="your_key_here"
```

## Pipeline

```bash
uv run scripts/boundaries/standardize_subnational_admin_areas.py
uv run scripts/boundaries/standardize_national_admin_areas.py
uv run scripts/boundaries/standardize_it_and_pa_areas.py
uv run scripts/boundaries/preprocess_mining_areas_and_query_calculator.py
```
