# Scalable Online Change Detection for High-dimensional Data Streams

This repository hosts the code for the paper:

- Maximum Mean Discrepancy on Exponential
Windows for Streaming Change Detection. (Currently under review).

We use publicly available data and release our code with an AGPLv3 license. If you are using code from this repository, please cite our paper.

# Running

To run the algorithm:

    install ChangeDetectors
    git clone <this-repository>
    cd <this-repository>
    conda create -n mmdew python=3.8 matplotlib seaborn numpy=1.20 scikit-learn jupyter tensorflow-cpu keras
    pip install . # tested with python 3.8
    python run_detectors.py
    

To produce the figures:


   # same as above
   python eval_results.py
   python eval_results_percent_changes_detected.py
   

# Results

![Main Results](figures/results.png)

![PCD and MTD](figures/percent_changes_detected.png)