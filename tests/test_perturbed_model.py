import unittest

from veriform.data.loaders import GSM8KLoader, ProcessBenchLoader
from veriform.preprocessing.dag import DAGModel
from veriform.perturbation.perturbers import StandardPerturber
from pprint import pprint

class TestPerturbedModel(unittest.TestCase):
    
    def test_perturbation(self):
        processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1000)
        chains = processbench.load()
        dag = DAGModel(chains[900])
        perturber = StandardPerturber(
            p=1, 
            operator_swap=True, 
            value_change=True, 
            logical_negation=True)
        
        dag = perturber.perturb(dag)

        for node in dag:
            print(node.contextualized())

            print(f"Original node content: {node.content}")
            print(f"Perturbed node content: {node.perturbed_content}")

# Run the tests
# python -m unittest tests.test_perturbed_model.TestPerturbedModel.test_perturbation