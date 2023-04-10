# Localise to segment: An investigation into the accuracy improvements provided by localisation steps for semantic segmentation.

Install dependencies.

    pip install -r requirements.txt

Installing textfig is more manual as no pip package is available.

    bash install_texfig.sh

The plots (optional) also require latex to be installed. 

    sudo apt-get install texlive-xetex   


Download the datasets from the Medical Segmentation Decathlon:

    bash download_datasets.sh
    
To prepare the other datasets (cropped and resized versions with 50 random samples).

    bach data_prep.sh 


To train the networks (Will likely take a couple of days depending on hardware)
    
    bash run_train.sh    
    

To compute results as csv and run t-test

    bash results.py


To make plots

    python make_plot_and_tables.py

