import numpy as np

def save(predictions):
	print('Save Submission')

	np.savetxt('../submissions/predictions.csv', predictions, delimiter=',')
