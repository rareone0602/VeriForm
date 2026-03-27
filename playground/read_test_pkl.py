import pickle
from pprint import pprint

pkl_file = 'tests/pkls/proven_perturbed_processbench_dag.pkl'

with open(pkl_file, 'rb') as f:
    formalized_dag = pickle.load(f)
for node in formalized_dag.nodes:
    print(f"node.id: {node.node_id}")
    print(f"node.flag: {node.flag}")
    print(f"node.is_perturbed: {node.is_perturbed}")
    print(f"node.content: {node.content}")
    print(f"node.perturbed_content: {node.perturbed_content}")
    print(f"node.formalized_content: \n```\n{node.formalized_content}\n```")
    print("--------------------------------------------------")