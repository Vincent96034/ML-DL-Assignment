## ML&DL @ CBS - Recommendation System
- Freese, Julius <jufr23ac@student.cbs.dk>
- Haunberger, Vincent <viha23al@student.cbs.dk>
- Schröder, Fynn <fysc23ab@student.cbs.dk>

This repository contains all code files for the final assignment of the course 'Machine Learning & Deep Learning' at Copenhagen Business School. The respective data files are too large to upload into this repository - they can be found in the submitted .zip file.

## How to run
Unfortunately the input datasets are too large to be uploaded to github. Instead we uploaded this entire project to [dropbox](https://www.dropbox.com/scl/fo/vy280k9kzdypda1ddfeuv/AGX7Eqa3XqatbAWewF20VxY?rlkey=pflywmcxn63er9wgaeehjvelv&st=e5arhvl3&dl=0) as well. To run, clone this repo and pull the data folder into the root directory. The data folder should have the following structure:
```
data
└───danish_stopwords.txt
|
└───embeddings
│   │   oai_embeddings_df.csv
│   │   tfidf_embeddings_df.csv
│ 
└───mappings
│   │   cid_mapping.csv
│   │   eid_lookup_dev.csv
│   │   eid_lookup_test.csv
│   │   uid_mapping.csv
│   
└───processed
│   │   events.csv
│   │   interactions.csv
│   │   items.csv
│   │   response_evaluation.pkl
│   │   user_embeddings.csv
│   │   users.csv
|
└───raw
    │   content_data.csv
    │   events_data.csv
```


## Structure

This repository is structured into 3 main notebooks. Additionally, there is the notebook `data_ops.ipynb` that includes all the data operations. Most Python functions and classes have been moved out of notebooks into dedicated `.py` files. Those can be found in the `common` module.

## Models
- Content-based Models: `CB_model.ipynb`
- Collaborative filtering: `CF_model.ipynb`
- Two Tower Neural Network: `2TM_model.ipynb`

