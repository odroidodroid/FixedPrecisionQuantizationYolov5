# 
from numpy import unsignedinteger
import torch
import copy
from utils.quant_utils.new_quant_utils import reduce_axis_max
import numpy as np
from scipy.stats import entropy
from collections import Counter


# 
def collect_stats(model, dataset, iter_num) :
    for index, image, _, _, _, _ in enumerate(dataset) :
        model(image.cudat())
        if index >= iter_num :
            break


class MaxCalibrator() :
    def __init__(self, bit_width, axis, unsigned, track_axis_max=False) :
        super(MaxCalibrator).__init__()
        self.track_axis_max = track_axis_max
        if self.track_axis_max : 
            self.axis_max = []
        self.calib_axis_max = None 
        self.axis = axis
        self.bit_width = bit_width
        self.unsigned  = unsigned

    @property
    def axis_max(self) :
        return self.axis_max

    def collect(self, x) :
        '''
            x is a tensor
        '''
        if torch.min(x) < 0 :
            x = x.abs()
        
        axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis] 
        reduce_axis = []

        for i in range(x.dim()) :
            if not i in axis :
                reduce_axis.append(i)

        local_axis_max = reduce_axis_max(x, axis=reduce_axis).detach()

        if self.calib_axis_max is None :
            self.calib_axis_max = local_axis_max

        else :
            if local_axis_max != self.calib_axis_max :
                assert('error! shape not matched')
            self.calib_axis_max = copy(torch.max(self.calib_axis_max, local_axis_max).data)
            
        if self.track_axis_max :
            self.axis_max.append(local_axis_max.cpu().numpy())        

    def reset(self) :
        self.calib_axis_max = None

    def return_axis_max(self) :
        return self.calib_axis_max
    



class HistogramCalibrator():
    def __init__(self, num_bits, axis, unsigned, num_bins=2048, grow_method=None, skip_zeros=False, torch_hist=False):
        super(HistogramCalibrator, self).__init__(num_bits, axis, unsigned)
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros

        self._calib_bin_edges = None
        self._calib_hist = None

        self._torch_hist = torch_hist

        if axis is not None:
            raise NotImplementedError("Calibrator histogram collection only supports per tensor scaling")


    def collect(self, x):
        if torch.min(x) < 0.:
            x = x.abs()

        x = x.float()

        if not self._torch_hist:
            x_np = x.cpu().detach().numpy()

            if self._skip_zeros:
                x_np = x_np[np.where(x_np != 0)]

            if self._calib_bin_edges is None and self._calib_hist is None:
                # first time it uses num_bins to compute histogram.
                self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)
            else:
                temp_amax = np.max(x_np)
                if temp_amax > self._calib_bin_edges[-1]:
                    # increase the number of bins
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(self._calib_bin_edges[-1] + width, temp_amax + width, width)
                    self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_bin_edges))
                hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
                hist[:len(self._calib_hist)] += self._calib_hist
                self._calib_hist = hist
        else:
            # This branch of code is designed to match numpy version as close as possible
            with torch.no_grad():
                if self._skip_zeros:
                    x = x[torch.where(x != 0)]

                # Because we collect histogram on absolute value, setting min=0 simplifying the rare case where
                # minimum value is not exactly 0 and first batch collected has larger min value than later batches
                x_max = x.max()
                if self._calib_bin_edges is None and self._calib_hist is None:
                    self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max)
                    self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
                else:
                    if x_max > self._calib_bin_edges[-1]:
                        width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                        self._num_bins = int((x_max / width).ceil().item())
                        self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

                    hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
                    hist[:self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist

    def reset(self):
        """Reset the collected histogram"""
        self._calib_bin_edges = None
        self._calib_hist = None

    def compute_axis_max(
            self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
        if isinstance(self._calib_hist, torch.Tensor):
            calib_hist = self._calib_hist.int().cpu().numpy()
            calib_bin_edges = self._calib_bin_edges.cpu().numpy()
        else:
            calib_hist = self._calib_hist
            calib_bin_edges = self._calib_bin_edges

        if method == 'entropy':
            calib_axis_max = _compute_axis_max_entropy(
                calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
        elif method == 'mse':
            calib_axis_max = _compute_axis_max_mse(
                calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
        elif method == 'percentile':
            calib_axis_max = _compute_axis_max_percentile(calib_hist, calib_bin_edges, percentile)
        else:
            raise TypeError("Unknown calibration method {}".format(method))

        return calib_axis_max

    # pylint:disable=missing-docstring
    def __str__(self):
        s = "HistogramCalibrator("
        if self._calib_bin_edges is None:
            bin_edge_str = "None"
        else:
            bin_edge_str = "[{:.3f}, ..., {:.3f}]({})".format(
                self._calib_bin_edges[0], self._calib_bin_edges[-1], len(self._calib_bin_edges))
        s += "calib_bin_edges={})".format(bin_edge_str)
        return s

    def __repr__(self):
        s = "HistogramCalibrator("
        s += super(HistogramCalibrator, self).__repr__()
        s += " calib_bin_edges={_calib_bin_edges}"
        s += " calib_hist={_calib_hist})"
        return s.format(**self.__dict__)
    # pylint:enable=missing-docstring


# Ideally, we want to decouple collector (collect histogram) and calibrator (compute amax) as opposed to
# the current calibrator design. The following compute amax functions are broken out from the calibrator
# as first step towards there.
def _compute_axis_max_entropy(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    def _normalize_distr(distr):
        summ = np.sum(distr)
        if summ != 0:
            distr = distr / summ

    bins = calib_hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    # we are quantizing to 128 values + sign if num_bits=8
    nbins = 1 << (num_bits - 1 + int(unsigned))

    starting = start_bin
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        _normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_axis_max = calib_bin_edges[last_argmin * stride + starting]
    calib_axis_max = torch.tensor(calib_axis_max.item()) #pylint: disable=not-callable

    return calib_axis_max

def _compute_axis_max_mse(calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128):
    """Returns amax that minimizes MSE of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    counts = torch.from_numpy(calib_hist[:]).float()
    edges = torch.from_numpy(calib_bin_edges[:]).float()
    centers = (edges[1:] + edges[:-1]) / 2

    mses = []
    arguments = []

    for i in range(start_bin, len(centers), stride):

        axis_max = centers[i]
        quant_centers = fake_tensor_quant(centers, axis_max, num_bits, unsigned)

        mse = ((quant_centers - centers)**2 * counts).mean()

        mses.append(mse)
        arguments.append(i)

    argmin = np.argmin(mses)
    calib_axis_max = centers[arguments[argmin]]

    return calib_axis_max

def _compute_axis_max_percentile(calib_hist, calib_bin_edges, percentile):
    """Returns amax that clips the percentile fraction of collected data"""

    if percentile < 0 or percentile > 100:
        raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    total = calib_hist.sum()
    cdf = np.cumsum(calib_hist / total)
    idx = np.searchsorted(cdf, percentile / 100)
    calib_axis_max = calib_bin_edges[idx]
    calib_axis_max = torch.tensor(calib_axis_max.item()) #pylint: disable=not-callable

    return calib_axis_max

def calibrate_weights(model, method="percentile", perchannel=True, percentile=99.99, num_bins=2048):
    for name, module in model.named_modules():
        if hasattr(module, ""):
            num_bits = module.weight_quantizer.num_bits
            unsigned = module.weight_quantizer.unsigned
            # channel_second_modules = (
            #     quant_nn.QuantConvTranspose1d,
            #     quant_nn.QuantConvTranspose2d,
            #     quant_nn.QuantConvTranspose3d
            # )
            if perchannel:
                axis = 1 if isinstance(module, channel_second_modules) else 0
            else:
                axis = None
            axis_size = module.weight.shape[axis] if axis is not None else 1

            # Histogram is always collected even if method is "max". Although "max" is supported here
            # but it is not the primary usage of this function
            if axis is None:
                calib_hist, calib_bin_edges = np.histogram(module.weight.abs().cpu().detach().numpy(), bins=2048)
                calib_hist = [calib_hist]
                calib_bin_edges = [calib_bin_edges]
            else:
                calib_hist = []
                calib_bin_edges = []
                for i in range(axis_size):
                    hist, bin_edges = np.histogram(
                        module.weight.index_select(
                            axis, torch.tensor(i, device=module.weight.device)).abs().cpu().detach().numpy(),
                        bins=num_bins)
                    calib_hist.append(hist)
                    calib_bin_edges.append(bin_edges)

            calib_axis_max = []
            if method == "max":
                reduce_axis = list(range(module.weight.dim()))
                reduce_axis.remove(axis)
                calib_axis_max.append(reduce_axis_max(module.weight, axis=reduce_axis))
            elif method == 'mse':
                for i in range(axis_size):
                    calib_axis_max.append(_compute_axis_max_mse(calib_hist[i], calib_bin_edges[i], num_bits, unsigned))
            elif method == 'percentile':
                for i in range(axis_size):
                    calib_axis_max.append(_compute_axis_max_percentile(calib_hist[i], calib_bin_edges[i], percentile))
            else:
                raise TypeError("Unsupported calibration method {}".format(method))

            if axis is None:
                calib_axis_max = calib_axis_max[0]
            else:
                calib_axis_max_shape = [1] * module.weight.dim()
                calib_axis_max_shape[axis] = module.weight.shape[axis]
                calib_axis_max = torch.stack(calib_axis_max).reshape(calib_axis_max_shape)
            module.weight_quantizer.axis_max = calib_axis_max.detach().cpu().numpy()