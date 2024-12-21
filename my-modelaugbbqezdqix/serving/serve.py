from typing import List
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn
import mlflow
import torch
import sys
import sentencepiece as spm
import os

# add parent directory to path
# add to original cookiecutter if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.nmt_model import NMT, Hypothesis

import mlflow
import mlflow.pytorch

# List of required environment variables
REQUIRED_ENV_VARS = ["MLFLOW_RUN_ID", "AZURE_STORAGE_CONNECTION_STRING", "MLFLOW_TRACKING_URI"]

# Check if all required variables are set
missing_vars = [var for var in REQUIRED_ENV_VARS if var not in os.environ]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# If all required variables are set, proceed
print("All required environment variables are set.")

app = FastAPI()

# Get the current working directory
current_dir = os.getcwd()

# Move two directories up
spm_src_path = f"runs:/{os.environ['MLFLOW_RUN_ID']}/spm_src.pkl"  # Adjust path to where you stored it

local_spm_src_path_path = mlflow.artifacts.download_artifacts(spm_src_path)

sp = joblib.load(local_spm_src_path_path)  

model_uri = f"runs:/{os.environ['MLFLOW_RUN_ID']}/models"
model = mlflow.pytorch.load_model(model_uri, map_location="cpu")

model.eval()

class PredictRequest(BaseModel):
    input_txt: str
    decoding_step: int
    beam_size: int
    
def beam_search(model: NMT, src_sent: List[str], beam_size: int, max_decoding_time_step: int) -> List[Hypothesis]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    model.eval()

    hypotheses = []
    with torch.no_grad():            
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
        hypotheses.append(example_hyps)

    return hypotheses


@app.post('/predict')
def predict(request: PredictRequest, response_model=List[Hypothesis]):
    input_txt = request.input_txt
    
    subword_tokens = sp.encode_as_pieces(input_txt)
    
    hypotheses = beam_search(model, subword_tokens,
                             beam_size=request.beam_size,
                             max_decoding_time_step=request.decoding_step)
    
    return hypotheses
    
@app.get("/")
def root():
    return {"message": "Hello world"}

def main():
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['LISTEN_PORT']))
    
if __name__ == "__main__":
    main()