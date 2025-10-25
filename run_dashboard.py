#!/usr/bin/env python
"""Launcher script for the Streamlit dashboard."""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run streamlit
if __name__ == '__main__':
    import streamlit.web.cli as stcli
    
    dashboard_path = os.path.join(project_root, 'dashboard', 'app.py')
    sys.argv = ['streamlit', 'run', dashboard_path]
    
    sys.exit(stcli.main())
