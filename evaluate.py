import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.spatial import distance
from detect import FaceEmbedder


def evaluate_verification(pairs_file, embed_model_path, dist='cosine'):
	face_embedder = FaceEmbedder(embed_model_path)
	model_type = Path(embed_model_path).stem.split('_')[0]
	count = 0

	# Direct pair verification
	pair_scores, pair_labels = [], []
	for line in open(pairs_file):
		count += 1
		p1, p2, same = line.strip().split()
		p1, p2 = f'dataset/{p1}', f'dataset/{p2}'
		e1, e2 = face_embedder.embed_image(p1), face_embedder.embed_image(p2)
		score = 1 - distance.cosine(e1, e2) if dist == 'cosine' else -np.linalg.norm(e1 - e2)
		pair_scores.append(score)
		pair_labels.append(int(same))
		print(f'{dist}_{model_type} - Verifying pairs - {count}/8805')

	if pair_scores:
		fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
		pair_auc = auc(fpr, tpr)
		print(f'Direct Pair AUC ({dist}): {pair_auc:.3f}')
		
		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1)
		plt.plot(fpr, tpr, label=f'AUC = {pair_auc:.3f}')
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Direct Pair Verification')
		plt.legend()
		plt.tight_layout()
		plt.savefig(f'images/results/{dist}_{model_type}.png', dpi=300, bbox_inches='tight')
		plt.close()
		return pair_auc
	

if __name__ == '__main__':
	pairs_file = 'dataset/verification_pairs_val.txt'
	for model in ['classification', 'metric']:
		for dist in ['cosine', 'euclidean']:
			embedding_model = f'models/{model}_embedding.keras'
			pairs_file = 'dataset/verification_pairs_val.txt'
			auc_score = evaluate_verification(pairs_file, embedding_model, dist)