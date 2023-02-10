# A simple Python script to
# (1) generate the BERT model and
# (2) apply it to the database

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sqlite3 as db
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def loadDB(dbConnection):
    """ Load the training database file into a proper database format
        PubMed 200k RCT dataset
        https://github.com/Franck-Dernoncourt/pubmed-rct
        15ktrain.txt (from the 20k folder)"""

    # Open the file
    filename = '15ktrain.txt'
    file = open(filename, "r").read()

    # Split it by the seperator defined by the published model
    seperator = "###"
    trials = file.split(seperator)

    # Use pandas to push them into a SQL lite file
    pd.DataFrame(trials, columns=['abstract_text']).to_sql(
        name="clinical_trials",     # Table name
        con=dbConnection,          # Open DB connection
        if_exists="replace",           # Fail if table is already present
        index=True,                 # Write DF frame index as column
        index_label="id",           # Gives index column a columnname
        chunksize=1000              # Write this many rows at a time
    )


def main():
    """Load, calculate and insert back"""
    print("BEGIN")

    # Connect to DB
    dbConnection = db.connect("15ktrain.db")

    # Load the database (this only needs to be executed once...)
    # loadDB(dbConnection)
    # results = generate(dbConnection)
    # updateDB(dbConnection, results)

    print("END")


# Run program
main()
