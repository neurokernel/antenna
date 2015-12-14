from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pkgutil
cuda_src_or_trans = pkgutil.get_data('neurokernel', 'LPU/neurons/cuda/odortransduction.cu')
cuda_src_hhn = pkgutil.get_data('neurokernel', 'LPU/neurons/cuda/hhn.cu')

class OdorReceptor(BaseNeuron):
    def __init__(self, n_dict, I_trans, dt, debug=False, LPU_id=None):
	self.floatType = np.float32

	self.num_neurons = len(n_dict['id'])
	self.LPU_id = LPU_id
	#super(OdorReceptor, self).__init__(n_dict, spk, dt, debug, LPU_id)
	self.debug = debug

	self.dt = dt

	self.I_trans = I_trans

	# allocate GPU memory for OR transduction
	self.d_bLR    = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_aG     = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_cAMP   = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_Ca     = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_CaCaM  = garray.to_gpu(1e-5*np.ones((self.num_neurons,1), dtype=self.floatType))
	self.d_aCaMK  = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_IX     = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_inhcng = garray.to_gpu(np.ones((self.num_neurons,1), dtype=self.floatType))
	self.d_I_cng  = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_I_cl   = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_I_ncx  = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_I_leak = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	self.d_V      = garray.zeros((self.num_neurons,1), dtype=self.floatType)
	#self.I_trans  = garray.zeros((self.num_neurons,1), dtype=self.floatType)

	self.binding_factor = garray.to_gpu(np.asarray(n_dict['BF'], dtype=self.floatType))

	_num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i]
					for i in range(self.num_neurons)],
					dtype=np.int32).flatten()
	_num_dendrite = np.asarray([n_dict['num_dendrites_I'][i]
				for i in range(self.num_neurons)],
				dtype=np.int32).flatten()

	self._cum_num_dendrite = garray.to_gpu(np.concatenate((
				np.asarray([0,], dtype=np.int32),
				np.cumsum(_num_dendrite, dtype=np.int32))))
	self._cum_num_dendrite_cond = garray.to_gpu(np.concatenate((
				np.asarray([0,], dtype=np.int32),
				np.cumsum(_num_dendrite_cond,
					dtype=np.int32))))
	self._num_dendrite = garray.to_gpu(_num_dendrite)
	self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
	self._pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
	self._cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
						dtype=np.int32))
	self._V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
					dtype=np.double))
	self.I = garray.zeros(self.num_neurons, np.double)
	self._update_I_cond = self._get_update_I_cond_func()
	self._update_I_non_cond = self._get_update_I_non_cond_func()

	self.update_ort = self._get_olfactory_transduction_kernel()
	self.update_hhn = self._get_multi_step_hhn_kernel()



    @property
    def neuron_class(self): return True

    def eval(self, st=None):

	self.update_ort.prepared_async_call(
		self.grid,
		self.block,
		st,
		self.num_neurons,
		self.dt,
		self.I.gpudata,
		self.binding_factor.gpudata,
		self.d_bLR.gpudata,
		self.d_aG.gpudata,
		self.d_cAMP.gpudata,
		self.d_Ca.gpudata,
		self.d_CaCaM.gpudata,
		self.d_aCaMK.gpudata,
		self.d_IX.gpudata,
		self.d_inhcng.gpudata,
		self.d_I_cng.gpudata,
		self.d_I_cl.gpudata,
		self.d_I_ncx.gpudata,
		self.d_I_leak.gpudata,
		self.d_V.gpudata,
		self.I_trans.gpudata)

    @property
    def update_I_override(self): return True

    def _not_called(self):
	self.update_hhn.prepared_async_call(
		self.grid,
		self.block,
		st,
		self.spk,
		self.num_neurons,
		self.dt*1000,
		self.I_drive.gpudata,
		self.X_1.gpudata,
		self.X_2.gpudata,
		self.X_3.gpudata,
		self.V.gpudata,
		self.V_prev.gpudata)

        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))
            self.V_file.root.array.append(self.V.get().reshape((1,-1)))

    def _get_olfactory_transduction_kernel(self):
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule(
		cuda_src_or_trans % { "type": dtype_to_ctype(self.floatType)},
		options=["--ptxas-options=-v"])

	func = mod.get_function("dougherty_transduction")
	func.prepare([
		np.int32,	# number of neurons
		self.floatType,	# dt
		self.floatType, #np.intp,	# Ostim/I
		np.intp,	# binding_factor
		np.intp,	# bLR,
		np.intp,	# aG,
		np.intp,	# cAMP,
		np.intp,	# Ca,
		np.intp,	# CaCaM,
		np.intp,	# aCaMK,
		np.intp,	# IX,
		np.intp,	# inhcng,
		np.intp,	# I_cng,
		np.intp,	# I_cl,
		np.intp,	# I_ncx,
		np.intp,	# I_leak,
		np.intp,	# V
		np.intp])	# I_trans
	return func


    def _get_hhn_kernel(self):
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule(
		cuda_src_hhn % {"type": dtype_to_ctype(self.floatType)},
		options=["--ptxas-options=-v"])
	func = mod.get_function("hhn_model")

	func.prepare([np.intp,       # spk
		      np.int32,      # num_neurons
		      self.floatType,     # dt
		      np.intp,        # I_pre
		      np.intp,        # X1
		      np.intp,        # X2
		      np.intp,        # X3
		      np.intp,        # g_V
		      np.intp])       # V_pre

	return func

    def _get_multi_step_hhn_kernel(self):
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule(
		cuda_src_hhn % {"type": dtype_to_ctype(self.floatType)},
		options=["--ptxas-options=-v"])
	func = mod.get_function("hhn_model_multi")

	func.prepare([np.intp,       # spk
		      np.int32,      # num_neurons
		      self.floatType,     # dt
		      np.intp,        # I_pre
		      np.intp,        # X1
		      np.intp,        # X2
		      np.intp,        # X3
		      np.intp,        # g_V
		      np.intp])       # V_pre

	return func
