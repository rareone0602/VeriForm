import json
from pathlib import Path

exp_dir = 'experiments/outputs'


def matrix_summary(confusion_matrix):
    total = sum(confusion_matrix.values())
    assert total > 0, "Total count in confusion matrix should be greater than zero."
    recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN']) if (confusion_matrix['TP'] + confusion_matrix['FN']) > 0 else 0
    precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP']) if (confusion_matrix['TP'] + confusion_matrix['FP']) > 0 else 0
    accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / total
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Total Samples: {total}")
    print("Confusion Matrix Summary:")
    print("-" * 40)
    print(f"  True Positives ratio: {confusion_matrix['TP']} ({(confusion_matrix['TP']/total)*100:.2f}%)")
    print(f"  False Positives ratio: {confusion_matrix['FP']} ({(confusion_matrix['FP']/total)*100:.2f}%)")
    print(f"  True Negatives ratio: {confusion_matrix['TN']} ({(confusion_matrix['TN']/total)*100:.2f}%)")
    print(f"  False Negatives ratio: {confusion_matrix['FN']} ({(confusion_matrix['FN']/total)*100:.2f}%)")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1_score*100:.2f}%")
    print("-" * 80)


def dir_key(name):
    _, _, _, _, p_str, n_str, b_str = name.split('_')
    p = int(p_str[1:])  # Extract percentage value
    n = int(n_str[1:])  # Extract number of samples
    b = int(b_str[1:])  # Extract batch size
    return (p, b)

for subdir in sorted(Path(exp_dir).iterdir(), key=lambda x: dir_key(x.name)):
    if subdir.is_dir():
        json_file = subdir / 'exp_results.json'
        if not json_file.exists():
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        p = data['p']
        prover_batch_size = data.get('prover_batch_size', 'N/A')
        confusion_matrix = {
            'TP': sum(item['true_positives'] for item in data['results']),
            'FP': sum(item['false_positives'] for item in data['results']),
            'TN': sum(item['true_negatives'] for item in data['results']),
            'FN': sum(item['false_negatives'] for item in data['results']),
        }

        print("#" * 80)
        print(f"Results for perturbation probability p={p}, prover batch size={prover_batch_size}:")
        matrix_summary(confusion_matrix)
