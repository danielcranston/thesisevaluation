import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def create_category_mask(wta, gt):
	"""
	wta and gt must be positive and pixel value 0 indicating missing values
	"""
	assert(wta.mean() > 0)
	assert(gt.mean() > 0)

	valids = gt != 0	# Found by GT
	founds = wta != 0	# Found by Saab Initial

	epe_wta = valids * founds * (np.abs(np.abs(wta) - np.abs(gt)))

	comb = np.zeros_like(valids, dtype=int)
	comb[epe_wta < 4 * valids] = 1			# Saabs Correct Predictions
	comb[epe_wta > 4] = 2					# Saabs Incorrect Prediction
	comb[np.invert(founds) * valids] = 3	# Saabs Missing Predictions (Holes)

	cmap = matplotlib.colors.ListedColormap(['gray', 'blue', 'red','green'])

	return comb, cmap

def calc_MAE(epe, mask):
	return np.abs(epe*mask).sum() / mask.sum()

def calc_RMSE(epe, mask):
	return np.sqrt(((epe*mask)**2).sum() / mask.sum())

def calc_metrics(epe_out, comb):
	valids = comb > 0

	mae_all = calc_MAE(epe_out, valids)
	mae1 = calc_MAE(epe_out, comb == 1)
	mae2 = calc_MAE(epe_out, comb == 2)
	mae3 = calc_MAE(epe_out, comb == 3)

	rmse_all = calc_RMSE(epe_out, valids)
	rmse1 = calc_RMSE(epe_out, comb == 1)
	rmse2 = calc_RMSE(epe_out, comb == 2)
	rmse3 = calc_RMSE(epe_out, comb == 3)

	bad4 = (epe_out > 4 * valids).sum() / valids.sum()
	bad2 = (epe_out > 2 * valids).sum() / valids.sum()

	mae = (mae_all, mae1, mae2, mae3)
	rmse = (rmse_all, rmse1, rmse2, rmse3)
	return mae, rmse, bad4, bad2
