# High-Throughput Machine Learning Screening of Double Perovskites 

# Guide

- You can reproduce part of our work with Juypter Notebook files in `scripts/`.

    - `00_generate_DP.ipynb` for DP bulk structures generation.
      
    - `01_calculate_stability.ipynb` for Pourbaix decomposition energy and energy above hull calculations from trajectories with DFT calculated energy.
 
    - `02_predict_stability.ipynb` for fingerprinting and ML predictions.
 
    - `03_generate_surface.ipynb` for DP surface structures generation.

- Generated and calculated DP bulk structures can be found in `data/`.

- Details of Gemnet-OC model for binding Gibbs free energy prediction can be found in `Gemnet-OC_for_DP/`.
