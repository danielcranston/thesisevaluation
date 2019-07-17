import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

def show_plots(wta, gt, output, epe_out, rgb, comb, cmap, epethresh, force_bads=True, show_now=True):
	
	minVal = min(wta.min(), output.min(), gt.min())
	maxVal = max(wta.max(), output.max(), gt.max())
	if force_bads:
		missing_in_gt = gt == 0
		missing_in_wta = wta == 0
		gt = gt.copy();				gt[missing_in_gt] = np.inf
		epe_out = epe_out.copy();	epe_out[missing_in_gt] = np.inf
		wta = wta.copy();			wta[missing_in_wta] = np.inf
		cmap2 = matplotlib.cm.viridis
		cmap2.set_bad('gray', 1.)
	else:
		cmap2 = matplotlib.cm.viridis

	plt.figure('wta'); plt.imshow(wta, cmap=cmap2, vmin=minVal, vmax=maxVal); plt.axis('off')
	plt.figure('gt'); plt.imshow(gt, cmap=cmap2, vmin=minVal, vmax=maxVal); plt.axis('off')
	plt.figure('output'); plt.imshow(output, cmap=cmap2, vmin=minVal, vmax=maxVal); plt.axis('off')
	plt.figure('rgb'); plt.imshow(rgb); plt.axis('off')
	plt.figure('comb'); plt.imshow(comb, cmap=cmap); plt.axis('off')
	plt.figure('epe_out{}'.format(epethresh)); plt.imshow(epe_out, cmap=cmap2, vmin=0, vmax=epethresh); plt.axis('off')
	if show_now:
		plt.show()

def show_overview(wta, gt, output, epe_out, rgb, comb, cmap, epethresh, title='', show_now=True):
	cmap_bads = matplotlib.cm.viridis
	cmap_bads.set_bad('gray', 1.)
	minVal = min(wta[wta!=0].min(), output[output!=0].min(), gt[gt!=0].min())
	maxVal = max(wta.max(), output.max(), gt.max())

	non_valids = gt == 0

	wta = np.abs(wta); wta[wta==0] = np.inf
	output = np.abs(output); output[output==0] = np.inf
	gt = np.abs(gt); gt[gt==0] = np.inf
	epe_out = np.abs(epe_out); epe_out[non_valids] = np.inf
	

	fig, axis = plt.subplots(2,3)
	fig.suptitle(title, fontsize=26)

	axis[0,0].imshow(wta, cmap=cmap_bads, vmin=minVal, vmax=maxVal)	; axis[0,0].title.set_text('input')
	axis[0,1].imshow(gt, vmin=minVal, vmax=maxVal)					; axis[0,1].title.set_text('gt')
	axis[0,2].imshow(output, vmin=minVal, vmax=maxVal)				; axis[0,2].title.set_text('output')
	axis[1,0].imshow(rgb)											; axis[1,0].title.set_text('rgb')
	axis[1,1].imshow(comb, cmap=cmap)								; axis[1,1].title.set_text('categories')
	axis[1,2].imshow(epe_out, vmin=0, vmax=epethresh)				; axis[1,2].title.set_text('EPE-{}'.format(epethresh))
	for ax in axis.ravel():
		ax.axis('off')
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()

	if show_now:
		fig.show()
	return fig

def show_epe_categories(epe, comb, epethresh, show_now=True):
	epe[epe == np.inf] = 0
	epe1 = epe * (comb == 1)
	epe2 = epe * (comb == 2)
	epe3 = epe * (comb == 3)
	epe1[epe1==0] = np.inf
	epe2[epe2==0] = np.inf
	epe3[epe3==0] = np.inf
	cmap2 = matplotlib.cm.viridis; cmap2.set_bad('gray', 1.)
	plt.figure('cat1_epe{}'.format(epethresh)); plt.imshow(epe1, cmap=cmap2, vmin=0, vmax=epethresh); plt.axis('off')
	plt.figure('cat2_epe{}'.format(epethresh)); plt.imshow(epe2, cmap=cmap2, vmin=0, vmax=epethresh); plt.axis('off')
	plt.figure('cat3_epe{}'.format(epethresh)); plt.imshow(epe3, cmap=cmap2, vmin=0, vmax=epethresh); plt.axis('off')
	plt.figure('all_epe{}'.format(epethresh)); plt.imshow(epe, cmap=cmap2, vmin=0, vmax=epethresh); plt.axis('off')
	if show_now:
		plt.show()



def get_result_string(mae, rmse, bad4, bad2, title):
	now = datetime.datetime.now()
	now = now.strftime('%Y-%m-%d %H:%M')

	string = ''
	string += '============================\n'
	string += 'Evaluation Results:\n'
	string += '{}\n'.format(now)
	string += '============================\n'
	string += title + '\n'
	string += '\n'
	string += 'MAE:\n'
	string += '  Category 1: {:.3f}\n'.format(mae[:,1].mean())
	string += '  Category 2: {:.3f}\n'.format(mae[:,2].mean())
	string += '  Category 3: {:.3f}\n'.format(mae[:,3].mean())
	string += '  Whole GT: {:.3f}\n'.format(mae[:,0].mean())
	string += '\n'
	string += 'RMSE:\n'
	string += '  Category 1: {:.3f}\n'.format(rmse[:,1].mean())
	string += '  Category 2: {:.3f}\n'.format(rmse[:,2].mean())
	string += '  Category 3: {:.3f}\n'.format(rmse[:,3].mean())
	string += '  Whole GT: {:.3f}\n'.format(rmse[:,0].mean())
	string += '\n'
	string += 'Bad4: {:.2f}%\n'.format(bad4.mean())
	string += 'Bad2: {:.2f}%\n'.format(bad2.mean())
	return string