## Install

`cd InDEP`

`pip install -r requirement.txt`

## Usage

`cd codes`

for example:

train the model

`python run.py -t PANCAN -m train -l InDEP`

get the score of genes

`python run.py -t PANCAN -m score -l InDEP`

test the model

`python run.py -t PANCAN -m eval -l InDEP`

get the interpretation of the model

`python run.py -t PANCAN -m interpretation -l InDEP`