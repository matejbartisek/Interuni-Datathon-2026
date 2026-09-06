# Interuni-Datathon-2026

# Gas Sensor Array Drift Compensation Pipeline

This repository contains the full automated pipeline for gas sensor classification across drifting batch environments.

## Repository Structure
* `requirements.txt`: Python package dependencies.
* `gas_sensor_pipeline.py`: Stage 1 base feature engineering, nested stacking meta-learner, and post-processing calibration script.
* `stage2_cascade_3_vs_5.py`: Stage 2 rescue cascade module for resolving Class 3/5 confusion using dedicated TGS2602 feature subsets.

## Execution
Run the pipeline stages sequentially from the root directory:

```bash
pip install -r requirements.txt
python gas_sensor_pipeline.py
python stage2_cascade_3_vs_5.py
```

submission_final.csv is a output from 1 STAGE: gas_sensor_pipeline.py

submission_final_cascaded.csv is a output from 2 STAGE: stage2_cascade_3_vs_5.py - THIS IS THE FINAL FINAL FINAL OUTPUT
