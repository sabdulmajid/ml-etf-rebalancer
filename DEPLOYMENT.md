# Deployment Guide

This document provides instructions for deploying the ML ETF Rebalancer application.

## Hosting Options

The application can be deployed in multiple ways:

### 1. Streamlit Cloud (Recommended)

The easiest way to host the dashboard:

1. Push your repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in and connect your GitHub account
4. Select this repository
5. Set the main file path to `dashboard/app.py`
6. Deploy!

Your app will be available at `https://[your-app-name].streamlit.app`

### 2. Vercel

For a more professional setup with a custom domain:

1. Push your repository to GitHub
2. Go to [Vercel](https://vercel.com)
3. Create a new project and import your repository
4. Configure the project:
   - Build command: leave empty
   - Output directory: leave empty
   - Install command: `pip install -r dashboard/requirements.txt`
   - Development command: leave empty
5. Deploy!

Your app will be available at `https://[your-project-name].vercel.app`

### 3. GitHub Pages

For a static version:

1. Run the application locally
2. Use a tool like `wget` to capture the HTML
3. Push to a GitHub Pages branch

### 4. Self-Hosting

For complete control:

1. Set up a server with Python
2. Clone the repository
3. Install dependencies: `pip install -r requirements.txt`
4. Run streamlit: `streamlit run dashboard/app.py`
5. Use nginx or similar to proxy requests

## Environment Variables

Make sure these environment variables are set in your hosting platform:

- `SECRET_KEY`: A random string for secure operations
- `PYTHON_VERSION`: Set to "3.10" or newer

## Automated Updates

The application is configured to automatically update via GitHub Actions on the first of each month. This ensures the portfolio is regularly rebalanced with fresh data and predictions.

To manually trigger a rebalance, you can run the workflow from the GitHub Actions tab in your repository.
