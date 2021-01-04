# Localise to segment: An investigation into the accuracy improvments provided by course-to-fine cascade and localisation steps for semantic segmentation of organs at risk.

Install dependencies.

    pip install -r requirements.txt

Installing textfig is more manual as no pip package is available.

    wget https://github.com/nilsleiffischer/texfig/archive/master.zip 
    unzip master.zip
    mv texfig-master/texfig.py ./
    mv master.zip -r texfig-master

The ThoracicOAR structseg dataset will need to be prepared. You need to download this yourself.

    Follow instruction at https://structseg2019.grand-challenge.org/Download
    And then moved into a folder named data/ThoracicOAR

To prepare the other datasets (cropped and resized versions of structseg).

    python data_prep.py
    

To train the networks (takes a long time)

    python train.py
    

To make plots

    python make_plots.py


To compute final results as csv and run t-test

    python compute_results.py
