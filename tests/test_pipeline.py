import unittest

from veriform.data.loaders import GSM8KLoader

from veriform.perturbation.perturbers import StandardPerturber
from veriform.formalization import GoedelFormalizer
from veriform.proving.deepseek_prover import DeepSeekProver
from veriform.pipeline import StandardPipeline



class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        perturber = StandardPerturber(
            p=0.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        formalizer = GoedelFormalizer()
        prover = DeepSeekProver()

        pipeline = StandardPipeline(perturber, formalizer, prover)

        dag = pipeline(chains[0])
        lean_program = dag.lean()
        print(lean_program)

        del pipeline

    def test_perturbed_pipeline(self):
        gsm8k = GSM8KLoader(split="test", num_samples=100)
        chains = gsm8k.load()
        perturber = StandardPerturber(
            p=1.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        formalizer = GoedelFormalizer()
        prover = DeepSeekProver()

        pipeline = StandardPipeline(perturber, formalizer, prover)

        dag = pipeline(chains[0])
        lean_program = dag.lean()
        print(lean_program)

        del pipeline

        
# Run the tests
# To run the test_pipeline only, run
# python -m unittest tests.test_pipeline.TestPipeline.test_pipeline
# To run the test_perturbed_pipeline only, run
# python -m unittest tests.test_pipeline.TestPipeline.test_perturbed_pipeline
# Run the full test_pipeline suite:
# python -m unittest tests.test_pipeline