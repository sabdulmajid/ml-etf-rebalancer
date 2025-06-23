from streamlit.web import cli as stcli
import os, sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.headless=true"]
    sys.exit(stcli.main())
