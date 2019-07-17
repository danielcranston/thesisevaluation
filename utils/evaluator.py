import numpy as np
import os
import imageio

from utils.metrics import create_category_mask, calc_metrics
from utils.visualization import show_plots, show_overview, get_result_string, show_epe_categories
from utils.io import readPFM
from utils.io import load_LIU_data, load_MiddV3_data, write_eval, scale_disp

class EvaluationItem():
	"""
	Container class, one for each individual item to be evaluated (Adiron, ArtL, ...)
	"""
	def __init__(self, name):
		self.name = name

		# Data
		self.wta = None
		self.gt = None
		self.output = None
		self.rgb = None

		# Results
		self.mae = None
		self.rmse = None
		self.bad4 = None
		self.bad2 = None
		self.epe_out = None
		self.details = ''

	def set_data(self, wta, gt, output, rgb):
		self.wta = wta
		self.gt = gt
		self.output = output
		self.rgb = rgb

	def set_results(self, mae, rmse, bad4, bad2, epe_out, details):
		self.mae = mae
		self.rmse = rmse
		self.bad4 = bad4
		self.bad2 = bad2
		self.epe_out = epe_out
		self.details = details

	def create_fig(self, epethresh=20):
		"""
		Shows a figure summarizing the data, output and resulting metrics for this item.
		"""
		comb, cmap = create_category_mask(self.wta, self.gt)
		fig = show_overview(self.wta, self.gt, self.output, self.epe_out, self.rgb, comb, cmap, epethresh, self.details, show_now=False)
		return fig

class Evaluator():
	def __init__(self, args):
		self.set = args.set
		self.dataset_name = 'LIU' if args.set == 'liu' else 'MiddV3'
		self.mode = args.mode
		self.plot = args.plot
		self.epethresh = args.epethresh

		# Filled as evaluator processes each item
		self.evalItems = []
		self.details_string = ''

		# Filled by compile_results()
		self.results_string = ''

		# Correctly setup items to evaluate and where to find them, depending on which dataset is used
		if args.set == 'liu':
			self.root = 'data/liu_dataset/'

			pairs = ['left_pair', 'middle_pair', 'right_pair']
			scenes = [str(i).zfill(3) for i in range(21,26)]
			self.items = [(pair, scene, False) for pair in pairs for scene in scenes] 	# 15 non merged
			self.items += [('middle_pair', scene, True) for scene in scenes]			# + the 5 merged ones
		elif args.set == 'middv3':
			self.root = 'data/MiddV3/trainingH/'

			self.items = sorted(next(os.walk(self.root))[1]) #Adiron, ArtL, ...
		else:
			raise Exception


	def load_data(self, item):
		"""
		Loads data correctly depending on which dataset (MiddV3/LIU) and method (nconv/sdr/inpaint/saab) is used
		"""
		if self.set == 'liu':
			pair, scene, bMerge = item
			wta, output, gt, rgb = load_LIU_data(self.root, pair, self.mode, scene)
			if bMerge:
				gt = imageio.imread(self.root + pair + '/left_gt_merged/' + scene + '.png')
				gt = np.abs(scale_disp(gt))
			return wta, output, gt, rgb

		if self.set == 'middv3':
			return load_MiddV3_data(self.root, item, self.mode)
		else:
			raise Exception

	def create_eval_item(self, item):
		"""
		Creates a container with appropriate name to store items data and results
		"""
		if self.set == 'liu':
			pair, scene, bMerge = item
			pair = pair + '(merge)' if bMerge else pair # middle_pair -> middle_pair(merge)
			return EvaluationItem(name='{:<18} - {}'.format(pair, scene))

		elif self.set == 'middv3':
			return EvaluationItem(name='{:<11}'.format(item))
		else:
			raise Exception

	def create_fig_path(self, item):
		"""
		Creates the path where an items figure should be saved
		"""
		if self.set == 'liu':
			pair, scene, bMerge = item
			scene = scene+'(merge)' if bMerge else scene
			folder =  'eval_results/LIU/{}'.format(pair)
			path   = folder + '/{}_{}.png'.format(self.mode.upper(), scene)
		elif self.set == 'middv3':
			folder = 'eval_results/MiddV3/' 
			path = folder + '/{}_{}.png'.format(self.mode.upper(), item)
		else:
			raise Exception

		os.makedirs(folder, exist_ok=True)
		return folder, path


	def process(self, wta, gt, output, name):
		"""
		Calculates results (metrics) for a given set of data
		"""
		valids = gt != 0

		comb, cmap = create_category_mask(wta, gt)
		epe_out = valids * np.abs(output - gt)

		mae, rmse, bad4, bad2 = calc_metrics(epe_out, comb)
		details = '{}: MAE={:6.3f}, RMSE={:6.3f}, BAD4={:5.2f}%, BAD2={:5.2f}%\n'.format(name, mae[0], rmse[0], 100*bad4, 100*bad2)
		self.details_string += details

		return mae, rmse, bad4, bad2, epe_out, details

	def compile_results(self):
		"""
		Once every evalItem has been processed individually, calculates result for entire dataset
		"""

		# highly performant code
		mae = np.array([ei.mae for ei in self.evalItems])
		rmse = np.array([ei.rmse for ei in self.evalItems])
		bad4 = 100 * np.array([ei.bad4 for ei in self.evalItems])
		bad2 = 100 * np.array([ei.bad2 for ei in self.evalItems])
		
		dataset_name = self.dataset_name + ' Dataset'
		self.results_string = get_result_string(mae, rmse, bad4, bad2, '{} - {}'.format(dataset_name, self.mode.upper()))
		print(self.results_string)
		
	def save_results(self):
		"""
		Once results have been compiled, saves them to text file
		"""
		root = 'eval_results/LIU/' if self.set == 'liu' else 'eval_results/MiddV3/'
		os.makedirs(root, exist_ok=True)

		file_string = root + 'eval_results_{}_{}.txt'.format(self.set, self.mode)
		write_eval(file_string, self.details_string, self.results_string)

