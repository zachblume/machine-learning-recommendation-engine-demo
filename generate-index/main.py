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


def loadTrialsTableFromTestData(dbConnection):
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


def generateTrialVectors(dbConnection):
    """Generate"""
    # Get MedBERT model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
    model = AutoModel.from_pretrained("Charangan/MedBERT")

    # Load DB to pandas dataframe
    trials = pd.read_sql_table(table_name='clinical_trials', con=dbConnection)

    # Get the text column and
    trialsList = trials.abstract_text.values.tolist()

    # Encode query and docs
    trialVectors = transform(trialsList)

    #


def pushVectorsToTrialTable(dbConnection):
    """Push the trial vectors back to DB"""


def dotProduct():
    # Trasnform text to vectors
    vectorOutput = transform(query)

    # Compute dot score between query and all trial vectors
    scores = torch.mm(vectorOutput, trialVectors.transpose(0, 1))[
        0].cpu().tolist()

    # Combine docs & scores
    doc_score_pairs = list(zip(docs, scores))

    # Sort by decreasing score
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    # Output passages & scores
    for doc, score in doc_score_pairs:
        print(score, doc)


def meanPooling(output, attentionMask):
    """Mean Pooling - Take average of all tokens
        This math was taken from a framework."""
    tokenVectors = output.last_hidden_state
    expandedMask = attentionMask.unsqueeze(
        -1).expand(tokenVectors.size()).float()
    return
    (torch.sum(tokenVectors * expandedMask, 1)
     /
     torch.clamp(expandedMask.sum(1), min=1e-9))


def transform(text, tokenizer, model):
    """Compute vectors from text"""

    # Tokenize
    tokenized = tokenizer(text, padding=True,
                          truncation=True, return_tensors='pt')

    # Transform, no_grad for memory saving
    with torch.no_grad():
        output = model(**tokenized, return_dict=True)

    # Pool
    vectors = meanPooling(output, tokenized['attention_mask'])

    # Normalize
    vectors = F.normalize(vectors, p=2, dim=1)

    # Return vectors (serialization later)
    return vectors


def main():
    """Load, calculate and insert back"""
    print("BEGIN")

    # Connect to DB
    dbConnection = db.connect("15ktrain.db")

    # Load the database (this only needs to be executed once...)
    # loadTrialsTableFromTestData(dbConnection)
    # results = generateTrialVectors(dbConnection)
    # pushVectorsToTrialTable(dbConnection, results)

    print("END")


# Run program
main()
