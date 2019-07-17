import numpy as np
import sys
import os

import glob
import imageio
import matplotlib.pyplot as plt
import matplotlib
import argparse

from utils.metrics import create_category_mask, calc_metrics
from utils.visualization import show_plots, show_overview, get_result_string, show_epe_categories
from utils.io import readPFM
from utils.io import load_LIU_data, load_MiddV3_data, write_eval, scale_disp
from utils.evaluator import Evaluator, EvaluationItem

parser = argparse.ArgumentParser()
parser.add_argument('-set', action='store',  type=str, dest='set', default=False, help='')
parser.add_argument('-mode', action='store', type=str, dest='mode', default='sdr', help='')
parser.add_argument('-plot', action='store', type=int,	 dest='plot', default=0, help='')
parser.add_argument('-epethresh', action='store', type=int,	 dest='epethresh', default=20, help='')
args = parser.parse_args()

if args.set not in ['liu', 'middv3']:
	print('Example usage: python evaluate.py -set SET -mode MODE')
	print('where:')
	print('    SET=[liu, middv3]')
	print('    MODE=[nconv, sdr, inpaint, saab]')
	print('optional parameters:')
	print('    -plot : sets plot mode. 0=disabled, 1=enabled, 2=enabled and saves figures to disc.')
	print('    -epethresh : sets end-point-error threshold (in pixels). must be a positive value.')
	sys.exit()
assert(args.mode in ['nconv', 'sdr', 'inpaint', 'saab'])
assert(args.epethresh > 0)
assert(args.plot in [0, 1, 2])

if args.mode == 'nconv' and args.set != 'liu':
	raise Exception('NCONV results are only available for the LIU dataset.')

# Create evaluator for the specified mode (liu / middv3)
evaluator = Evaluator(args)

# Loop through every item in dataset
for item in evaluator.items:
	# Create container for this item, with appropriate name (Adiron, ArtL, ...)
	evalItem = evaluator.create_eval_item(item)

	# Load this items data and assign to container
	wta, output, gt, rgb = evaluator.load_data(item)
	evalItem.set_data(wta, gt, output, rgb)

	# Let the evaluator process data, get this items metrics
	mae, rmse, bad4, bad2, epe_out, details = evaluator.process(wta, gt, output, evalItem.name)

	# Assign the results to the container
	evalItem.set_results(mae, rmse, bad4, bad2, epe_out, details)
	print('Processed', evalItem.details, end='')

	# Store the container in evaluator
	evaluator.evalItems.append(evalItem)

	# Present the result of this item
	if args.plot > 0:
		fig = evalItem.create_fig(epethresh=args.epethresh)
		plt.show()
		if args.plot > 1:
			folder, path = evaluator.create_fig_path(item)
			fig.savefig(path)
			plt.close(fig)


evaluator.compile_results() # Compile and present final results
evaluator.save_results()	# Save final results to text file
