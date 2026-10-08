import tensorflow as tf
from vegasflow import VegasFlow
from vegasflow.configflow import DTYPE, TECH_CUT

# --- VEGAS grid training ---
class VegasFlowImproved(VegasFlow):
    """VegasFlow with an O(N) fill of the grid-training histogram and, with jit_event=True,
    each chunk of events compiled as one XLA cluster.

    VegasFlow sums (w f)^2 per bin with a dense (bins x N) one-hot mask for every dimension
    (vegasflow.utils.consume_array_into_indices). Here all dimensions are filled with one
    segment sum over flattened (dimension, bin) indices. Same histogram up to summation order.

    VegasFlow runs each chunk (random numbers, grid mapping, integrand, sums, histogram) as
    separate TensorFlow ops. With jit_event=True the whole chunk is one XLA cluster. The random
    numbers then come from a tf.random.Generator, which works under XLA and advances on every
    call: seeded with `seed`, or from a non-deterministic state (a fresh stream each run) if None."""
    def __init__(self, n_dim, n_events, seed=None, jit_event=True, **kwargs):
        super().__init__(n_dim, n_events, **kwargs)
        self.jit_event = jit_event
        if seed is None:
            self._rng = tf.random.Generator.from_non_deterministic_state()
        else:
            # The seed is the Philox key, so different seeds give independent streams.
            # (Generator.from_seed puts the seed into the counter: seeds s and s+1 give the same
            # stream shifted by one step.)
            self._rng = tf.random.Generator.from_key_counter(key=seed, counter=[0, 0], alg="philox")

    def compile(self, integrand, compilable=True, signature=None, trace=False, check=True):
        super().compile(integrand, compilable=compilable, signature=signature, trace=trace, check=check)
        if compilable and self.jit_event:
            self.event = tf.function(self.event.python_function, jit_compile=True)

    def _generate_random_array(self, n_events, *args):
        # MonteCarloFlow._generate_random_array with the random numbers drawn from self._rng
        rnds_raw = self._rng.uniform((n_events, self.n_dim), minval=TECH_CUT, maxval=1.0 - TECH_CUT, dtype=DTYPE)
        rnds, wgts_raw, *extra = self._digest_random_generation(rnds_raw, *args)

        wgts = wgts_raw * self.xjac
        if self._xdelta is not None:
            rnds = self._xmin + rnds * self._xdelta
            wgts *= self._xdeltajac
        return rnds, wgts, *extra

    def _can_run_vectorial(self, expected_shape):
        # VegasFlow accepts vectorial (batched beta) integrands only when the class is named exactly
        # "VegasFlow". The histogram fill below receives only the main dimension, as in VegasFlow.
        super()._can_run_vectorial(expected_shape) # keeps the main_dimension range check
        return True

    def _importance_sampling_array_filling(self, results2, indices):
        if not self.train:
            return []

        n_bins = self.grid_bins - 1
        # indices: (N, n_dim) bin of each event in each dimension -> segment id dim*n_bins + bin
        segment_ids = indices + n_bins * tf.range(self.n_dim, dtype=indices.dtype)
        data = tf.broadcast_to(tf.expand_dims(results2, -1), tf.shape(indices))
        arr_res2 = tf.math.unsorted_segment_sum(data, segment_ids, self.n_dim * n_bins)

        return tf.reshape(arr_res2, (self.n_dim, n_bins))