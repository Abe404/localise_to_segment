# Download datasets
# Pancreas
# Prostate central gland
# Spleen 
# Left Atrium 


if [ ! -f data/spleen/dl.tar ]; then # if not already downloaded
    mkdir -p data/spleen # create folder if it does not exist
    gdown 1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE --output data/spleen/dl.tar
    cd data/spleen; tar -xvf dl.tar; cd ../..;
    # now the data is located at data/pancreas/Task09_Spleen
fi

if [ ! -f data/heart/dl.tar ]; then # if not already downloaded
    mkdir -p data/heart # create folder if it does not exist
    gdown 1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY --output data/heart/dl.tar
    cd data/heart; tar -xvf dl.tar; cd ../..;
    # now the data is located at data/pancreas/Task02_Heart
fi

if [ ! -f data/prostate/dl.tar ]; then # if not already downloaded
    mkdir -p data/prostate # create folder if it does not exist
    gdown 1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a --output data/prostate/dl.tar
    cd data/prostate; tar -xvf dl.tar; cd ../..;
fi

if [ ! -f data/pancreas/dl.tar ]; then # if not already downloaded
    mkdir -p data/pancreas # create folder if it does not exist
    gdown 1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL --output data/pancreas/dl.tar
    cd data/pancreas; tar -xvf dl.tar; cd ../..;
fi

if [ ! -f data/liver/dl.tar ]; then # if not already downloaded
    mkdir -p data/liver # create folder if it does not exist
    gdown 1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu --output data/liver/dl.tar
    cd data/liver; tar -xvf dl.tar; cd ../..;
fi
