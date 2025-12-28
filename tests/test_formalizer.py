import unittest

from veriform.data_collection.dataset_loaders import GSM8KLoader, ProcessBenchLoader
from veriform.autoformalization_v2.dag import StandardDAGModel
from veriform.autoformalization_v2.perturber import StandardPerturber
from veriform.autoformalization_v2.formalizer import StepfunFormalizer, KiminaFormalizer, GoedelFormalizer, HeraldFormalizer
import pickle

class BaseFormalizerTest(unittest.TestCase):
    """Base class for formalizer tests with common test methods."""
    
    formalizer_class = None  # Override in subclass
    
    def _run_formalization_test(self, dag, formalizer, output_file):
        """Common formalization test logic."""
        formalized_dag = formalizer.formalize(dag)
        for node_nl, node_fl in zip(dag.nodes, formalized_dag.nodes):
            print(f"Original node content: {node_nl.content}")
            print(f"Formalized node content: ```\n{node_fl.formalized_content}```")
        with open(output_file, "wb") as f:
            pickle.dump(formalized_dag, f)
        return formalized_dag
    
    def test_formalization_gsm8k(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        dag = StandardDAGModel(chains[0])
        formalizer = self.formalizer_class()
        self._run_formalization_test(dag, formalizer, "tests/pkls/formalized_dag.pkl")

    def test_formalization_perturbed_processbench(self):
        processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1)
        chains = processbench.load()
        dag = StandardDAGModel(chains[0])
        perturber = StandardPerturber(p=1, operator_swap=True, value_change=True, logical_negation=True)
        formalizer = self.formalizer_class()
        
        for i in range(len(dag)):
            dag = perturber.perturb(dag, step_id=i)
        
        self._run_formalization_test(dag, formalizer, "tests/pkls/formalized_perturbed_processbench_dag.pkl")

    def test_perturbed_formalization(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        dag = StandardDAGModel(chains[0])
        perturber = StandardPerturber(p=1.0, operator_swap=True, value_change=True, logical_negation=True)
        
        for step_id in range(3):
            dag = perturber.perturb(dag, step_id=step_id)
        
        formalizer = self.formalizer_class()
        self._run_formalization_test(dag, formalizer, "tests/pkls/formalized_perturbed_dag.pkl")


class TestStepfunFormalizer(BaseFormalizerTest):
    formalizer_class = StepfunFormalizer


class TestKiminaFormalizer(BaseFormalizerTest):
    formalizer_class = KiminaFormalizer


class TestGoedelFormalizer(BaseFormalizerTest):
    formalizer_class = GoedelFormalizer

class TestHeraldFormalizer(BaseFormalizerTest):
    formalizer_class = HeraldFormalizer

# Run the tests
# python -m unittest tests.test_formalizer

