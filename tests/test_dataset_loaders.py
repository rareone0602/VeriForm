import unittest
from veriform.data_collection.dataset_loaders import GSM8KLoader, MATHLoader, ProcessBenchLoader
from veriform.data_collection.reasoning_step import ReasoningChain
from veriform.autoformalization_v2.dag import DAGModel

class TestDatasetLoaders(unittest.TestCase):

    def test_gsm8k_loader(self):
        """Test GSM8KLoader with a small sample."""
        loader = GSM8KLoader(split="test", num_samples=2, seed=42)
        chains = loader.load()

        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        self.assertIsInstance(chains[0], ReasoningChain)
        self.assertEqual(chains[0].source_dataset, "gsm8k")

    def test_math_loader(self):
        """Test MATHLoader with a small sample."""
        loader = MATHLoader(split="test", num_samples=2, seed=42)
        chains = loader.load()

        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        self.assertIsInstance(chains[0], ReasoningChain)
        self.assertEqual(chains[0].source_dataset, "math")

    def test_processbench_loader(self):
        """Test ProcessBenchLoader with a small sample."""
        loader = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=2, seed=42)
        chains = loader.load()
        

        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        self.assertIsInstance(chains[0], ReasoningChain)
        
        dag_model = DAGModel(chains[1])
        for i in range(len(dag_model)):
            node = dag_model[i]
            if node.is_declarative:
                continue

            print('-' * 40)
            print(node.contextualized())
            print('-' * 40)
            print()

if __name__ == "__main__":
    unittest.main()

# python -m unittest tests.test_dataset_loaders.TestDatasetLoaders