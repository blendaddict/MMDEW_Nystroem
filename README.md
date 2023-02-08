# Maximum Mean Discrepancy on Exponential Windows for Streaming Change Detection

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
    python unify_results.py 1  ../results/<date> && python unify_results.py 4  ../results/<date>
    

Use the notebooks in `notebooks` to produce the figures.   

# Results

![Main Results](figures/results.png)

![PCD and MTD](figures/percent_changes_detected.png)

![Runtime](figures/runtime.png)