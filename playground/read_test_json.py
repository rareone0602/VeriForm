import json
from pprint import pprint

json_file = 'data/tmp/regex_perturbed.json'

with open(json_file, 'r') as f:
    data = json.load(f)

# Example access

formlizer = "goedel" # "kimina" | "herald" | "stepfun"
perturbation = "unperturbed" # "perturbed"
node_id = 5
chain_id = "olympiadbench-552"
pprint(data[formlizer][chain_id][node_id][perturbation])
