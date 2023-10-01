import logging

from pyhf import get_backend, events
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


def required_parset(sample_data, modifier_data):
    return {
        "paramset_type": "unconstrained",
        "n_parameters": 1,
        "is_scalar": True,
        "inits": (0.0,),
        "bounds": ((-10, 10),),
        "fixed": False,
    }


class eftlin_builder:
    """Builder class for collecting eftlin modifier data"""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        maskval = True if thismod else False
        mask = [maskval] * len(nom)
        return {"mask": mask}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            "data", {"mask": []}
        )
        nom = (
            defined_samp["data"]
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]["data"]["mask"] += moddata["mask"]
        if thismod:
            self.required_parsets.setdefault(
                thismod["name"],
                [required_parset(defined_samp["data"], thismod["data"])],
            )

    def finalize(self):
        return self.builder_data


class eftlin_combined:
    name = "eftlin"
    op_code = "multiplication"

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        self.batch_size = batch_size

        keys = [f"{mtype}/{m}" for m, mtype in modifiers]
        eftlin_mods = [m for m, _ in modifiers]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, eftlin_mods)

        self._eftlin_mask = [
            [[builder_data[m][s]["data"]["mask"]] for s in pdfconfig.samples]
            for m in keys
        ]
        self._precompute()
        events.subscribe("tensorlib_changed")(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.eftlin_mask = tensorlib.tile(
            tensorlib.astensor(self._eftlin_mask), (1, 1, self.batch_size or 1, 1)
        )
        self.eftlin_mask_bool = tensorlib.astensor(self.eftlin_mask, dtype="bool")
        self.eftlin_default = tensorlib.ones(self.eftlin_mask.shape)

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            eftlins = self.param_viewer.get(pars)
            results_eftlin = tensorlib.einsum("msab,m->msab", self.eftlin_mask, eftlins)
        else:
            eftlins = self.param_viewer.get(pars)
            results_eftlin = tensorlib.einsum(
                "msab,ma->msab", self.eftlin_mask, eftlins
            )

        results_eftlin = tensorlib.where(
            self.eftlin_mask_bool, results_eftlin, self.eftlin_default
        )
        return results_eftlin


class eftquad_builder:
    """Builder class for collecting eftquad modifier data"""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    def collect(self, thismod, nom):
        maskval = True if thismod else False
        mask = [maskval] * len(nom)
        return {"mask": mask}

    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            "data", {"mask": []}
        )
        nom = (
            defined_samp["data"]
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)
        self.builder_data[key][sample]["data"]["mask"] += moddata["mask"]
        if thismod:
            self.required_parsets.setdefault(
                thismod["name"],
                [required_parset(defined_samp["data"], thismod["data"])],
            )

    def finalize(self):
        return self.builder_data


class eftquad_combined:
    name = "eftquad"
    op_code = "multiplication"

    def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
        self.batch_size = batch_size

        keys = [f"{mtype}/{m}" for m, mtype in modifiers]
        eftquad_mods = [m for m, _ in modifiers]

        parfield_shape = (
            (self.batch_size, pdfconfig.npars)
            if self.batch_size
            else (pdfconfig.npars,)
        )
        self.param_viewer = ParamViewer(parfield_shape, pdfconfig.par_map, eftquad_mods)

        self._eftquad_mask = [
            [[builder_data[m][s]["data"]["mask"]] for s in pdfconfig.samples]
            for m in keys
        ]
        self._precompute()
        events.subscribe("tensorlib_changed")(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if not self.param_viewer.index_selection:
            return
        self.eftquad_mask = tensorlib.tile(
            tensorlib.astensor(self._eftquad_mask), (1, 1, self.batch_size or 1, 1)
        )
        self.eftquad_mask_bool = tensorlib.astensor(self.eftquad_mask, dtype="bool")
        self.eftquad_default = tensorlib.ones(self.eftquad_mask.shape)

    def apply(self, pars):
        """
        Returns:
            modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()
        if self.batch_size is None:
            eftquads = self.param_viewer.get(pars)
            eftquads = tensorlib.power(eftquads, 2)
            # print(eftquads)
            results_eftquad = tensorlib.einsum(
                "msab,m->msab", self.eftquad_mask, eftquads
            )
        else:
            eftquads = self.param_viewer.get(pars)
            eftquads = tensorlib.power(eftquads, 2)
            # print(eftquads)
            results_eftquad = tensorlib.einsum(
                "msab,ma->msab", self.eftquad_mask, eftquads
            )

        results_eftquad = tensorlib.where(
            self.eftquad_mask_bool, results_eftquad, self.eftquad_default
        )
        return results_eftquad
