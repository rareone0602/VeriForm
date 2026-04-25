from .data.reasoning_chain import ReasoningChain
from .preprocessing.dag import DAGModel
# --- Component Signatures (Protocols) ---
# These define what your helper objects must look like ("Duck Typing").
# It is 'spot on' for keeping dependencies loose.



# --- Main Pipeline Signature ---


class StandardPipeline:
    """
    The concrete implementation. This is where the rubber meets the road.
    """
    def __init__(self,
                perturbation_model, # Use the Protocol type!
                formalizer_model,
                prover_model):
        self.perturbation_model = perturbation_model
        self.formalizer_model = formalizer_model
        self.prover_model = prover_model
    
    def __call__(self, reasoning_chain: ReasoningChain) -> DAGModel:
        """Execute the full pipeline."""
        dag = DAGModel(reasoning_chain)

        dag = self.perturbation_model.perturb(dag)

        dag = self.formalizer_model.formalize(dag)

        dag = self.prover_model.prove(dag)

        return dag