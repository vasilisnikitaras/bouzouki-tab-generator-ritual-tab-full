#!/bin/bash
pip install -r requirements.txt
streamlit run ritual_tab_full.py --server.port=$PORT --server.address=0.0.0.0
