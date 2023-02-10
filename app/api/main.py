"""
 A simple Python script to:
 * Load the medBERT model
 * Produce and store vectors for 15k public clinical trial article abstracts
 * Produce and store vectors for a series of example clinical questions
 * Precalculate the best matching trial for each question
 * Allow for realtime search by the consumer appliaction by generalizing these methods
"""

from http.server import BaseHTTPRequestHandler
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sqlite3 as db
import random
import torch
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import codecs
import pickle


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
        chunksize=1000              # Write this many rows at a time to db
    )

    # Add the serialized vector column for later
    cursor = dbConnection.cursor()
    sql = """
            ALTER TABLE clinical_trials
            ADD COLUMN serialized_vectors text;
            """
    cursor.execute(sql)
    dbConnection.commit()


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
        print("TRIAL VECTORIZE LOOP "+str(i))

        # Chunk into 100s
        start = i*100
        end = start+100
        if (end > len(trialsList)):
            end = len(trialsList)-1
        chunk = trialsList[start:end]

        # Transform list of trials to list of trialVectors
        trialVectorsChunk = transform(chunk, tokenizer, model)

        # Add to output
        allTrialVectors += trialVectorsChunk

    return allTrialVectors


def pushVectorsToTrialTable(dbConnection, allTrialVectors):
    """Push the trial vectors back to DB"""

    # Prep db cursor
    cursor = dbConnection.cursor()

    # Prep SQL statement to push SQL UPDATEs
    sql = """
            UPDATE clinical_trials
            SET serialized_vectors = '{0}'
            WHERE ID = {1}
        """

    # Serialize every item in allTrialVectors and push update to db
    for i in range(len(allTrialVectors)):
        # Pickle and encode bytes to base64 string
        pickled = codecs.encode(pickle.dumps(
            allTrialVectors[i]), "base64").decode()

        # ID of the row we're updating
        id = i

        # Format the SQL command with the id & vector and then execute it on a cursor, and finally commit the transaction
        query = sql.format(pickled, id)
        cursor.execute(query)

    # Commit the transactions
    dbConnection.commit()


def findClosestDotProduct(text, tokenizer, model, dbConnection):
    # Transform text to vector embedding
    vectorOutputToCompare = transform(text, tokenizer, model)

    # Load trialVectors from database
    trialTable = pd.read_sql_query(
        "SELECT * from clinical_trials WHERE abstract_text!='' LIMIT 1000", dbConnection)

    # Panda=>list for trialIDs and serialized_vectors
    trialIDs = trialTable.id.values.tolist()
    trialVectors = trialTable.serialized_vectors.values.tolist()
    trialAbstracts = trialTable.abstract_text.values.tolist()

    # Deserialize (unpickle and base 64 decode) the vector embeddings in place
    for i in range(len(trialVectors)):
        trialVectors[i] = pickle.loads(codecs.decode(
            trialVectors[i].encode(), "base64"))

    # print(trialVectors[0])
    # return

    trialVectors = torch.stack(trialVectors)

    # Compute dot score between query and all trial vectors
    scores = torch.mm(vectorOutputToCompare, trialVectors.transpose(0, 1))[
        0].cpu().tolist()

    # Combine trialIDs & scores
    IDscorePairs = list(zip(trialIDs, scores))

    # Sort by decreasing score
    IDscorePairs = sorted(IDscorePairs, key=lambda x: x[1], reverse=True)

    # Output passages & scores
    results = []
    for ID, score in IDscorePairs:
        # return ID
        # return scores[]
        results.append(trialAbstracts[ID])
    return results


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

    # If the trials table doesn't exist...
    cursor = dbConnection.cursor()
    trialTableExists = cursor.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table'
        AND name='clinical_trials'
        """).fetchall()

    if not trialTableExists:
        # Precompute the trial vector embeddings and store in the database (this only needs to be executed once...)
        loadTrialsTableFromTestData(dbConnection)
        allTrialVectors = generateTrialVectors(dbConnection, tokenizer, model)
        pushVectorsToTrialTable(dbConnection, allTrialVectors)

    else:
        print("Table already precomputed")

        query = input("Text to match to?: ")
        result = findClosestDotProduct(query, tokenizer, model, dbConnection)
        print(result)

    print("END")


# Run program
# main()


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        query = "cancer"
        result = "hello world"
        # result = findClosestDotProduct(query, tokenizer, model, dbConnection)

        self.wfile.write(results.encode('utf-8'))
        return result
