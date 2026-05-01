# Deployment

Keep deployment simple: this project is a read-only Streamlit app backed by committed artifacts.

## Local

```bash
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Streamlit Community Cloud

1. Connect the GitHub repository.
2. Set the app entrypoint to `dashboard/app.py`.
3. Use the root `requirements.txt`.
4. Deploy.

The deployed app does not run model training from the UI. Refresh artifacts locally with `python run_pipeline.py`, review the outputs, commit them, and redeploy from GitHub.

## Refreshing Data

```bash
python run_pipeline.py
git add artifacts/latest
git commit -m "Refresh research artifacts"
```

This avoids fragile scheduled jobs and keeps every public result traceable to a reviewed commit.
