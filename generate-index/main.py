"""
 A simple Python script to:
 * Load the medBERT model
 * Produce and store vectors for 15k public clinical trial article abstracts
 * Produce and store vectors for a series of example clinical questions
 * Precalculate the best matching trial for each question
 * Allow for realtime search by the consumer appliaction by generalizing these methods
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sqlite3 as db
import random
import torch
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def loadTrialsTableFromTestData(dbConnection):
    """
    Load the training database file into a proper database format
    PubMed 200k RCT dataset
    https://github.com/Franck-Dernoncourt/pubmed-rct
    15ktrain.txt (from the 20k folder)
    """

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


def generateTrialVectors(dbConnection, tokenizer, model):
    """Generate embeddings for each of the trial db rows"""

    # Load DB to pandas dataframe
    # trials = pd.read_sql_table(table_name='clinical_trials', con=dbConnection) # Actually this is SQLalchemy
    trials = pd.read_sql_query(
        "SELECT * from clinical_trials WHERE abstract_text!='' LIMIT 1000", dbConnection)

    # Get the text column and transform it from a panda into a simple py list
    trialsList = trials.abstract_text.values.tolist()

    # Transform the text to vector embeddings using model
    # Let's make this maneagable lists of 100 at a time
    i = 0
    allTrialVectors = []
    for i in range(math.ceil(len(trialsList)/100)):
        # Mark the loop
        print("LOOP "+str(i))

        # Chunk into 100s
        start = i*100
        end = start+100
        if (end > len(trialsList)):
            end = len(trialsList)-1
        chunk = trialsList[start:end]

        # Transform list of trials to list of trialVectors
        trialVectorsChunk = transform(chunk, tokenizer, model)

        # Add to output
        allTrialVectors.append(trialVectorsChunk)

    return allTrialVectors


def pushVectorsToTrialTable(dbConnection, allTrialVectors):
    """Push the trial vectors back to DB"""

    # First we need to serialize allTrialVectors

    # Then we need to potentially add a column for serailizedVectors

    # Then we can push SQL UPDATEs
    sql = """
        UPDATE clinical_trials
        SET vectors = {0}
        WHERE ID = {1}
    """

    # Format the SQL command with the id & vector and then execute it on a cursor, and finally commit the transaction
    dbConnection.cursor().execute(sql.format(serializedVectors, id)).commit()


def dotProduct(text, trialVectors):
    # Transform text to vectors
    vectorOutput = transform(text)

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
    """
    Mean Pooling - Take average of all tokens
    This math was taken from a framework.
    """
    tokenVectors = output.last_hidden_state
    expandedMask = attentionMask.unsqueeze(
        -1).expand(tokenVectors.size()).float()
    return (
        torch.sum(tokenVectors * expandedMask, 1) /
        torch.clamp(expandedMask.sum(1), min=1e-9)
    )


def transform(text, tokenizer, model):
    """Compute vectors from text"""

    # BERT can only handle 512 tokens

    # Tokenize
    tokenized = tokenizer(text, padding=True,
                          # BERT CAN ONLY ACCEPT <512 tokens including special [CLS] and [SEP]
                          max_length=510,
                          truncation='longest_first',
                          return_tensors='pt')

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

    # Get MedBERT model and tokenizer (from HuggingFace database)
    # It also thankfully caches the .bin model file so no crazy traffic with nodemon hot reloading
    tokenizer = AutoTokenizer.from_pretrained("Charangan/MedBERT")
    model = AutoModel.from_pretrained("Charangan/MedBERT")

    # Connect to DB
    dbConnection = db.connect("15ktrain.sqlite")

    # Load the database (this only needs to be executed once...)
    loadTrialsTableFromTestData(dbConnection)
    allTrialVectors = generateTrialVectors(dbConnection, tokenizer, model)
    pushVectorsToTrialTable(dbConnection, allTrialVectors)

    print("END")


# Run program
main()
