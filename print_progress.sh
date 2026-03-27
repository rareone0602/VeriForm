shopt -s nullglob

date +%s

for model in goedel stepfun herald kimina; do
    files=(data/llm_perturbed_low/proved/${model}/100/*.json)
    echo "${model} low ${#files[@]}"
    files=(data/llm_perturbed_medium/proved/${model}/100/*.json)
    echo "${model} medium ${#files[@]}"
    files=(data/llm_perturbed_high/proved/${model}/100/*.json)
    echo "${model} high ${#files[@]}"
done
