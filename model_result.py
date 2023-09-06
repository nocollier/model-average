import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Self, Union

import numpy as np
import xarray as xr
import yaml

warnings.simplefilter("ignore", category=xr.SerializationWarning)


@dataclass
class ModelResult:
    """A class for abstracting and managing model results and ensembles."""

    name: str = field(init=True, default_factory=str)
    children: dict = field(init=False, default_factory=dict)
    synonyms: dict = field(init=False, repr=False, default_factory=dict)
    variables: dict = field(init=False, repr=False, default_factory=dict)
    area_atm: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)
    area_ocn: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)
    frac_lnd: xr.DataArray = field(init=False, repr=False, default_factory=lambda: None)

    def find_files(self, path: Union[str, list[str]]) -> Self:
        """Populate a database of variables found in netCDF files in the given paths.

        Parameters
        ----------
        path
            The location or list of locations in which to search for netCDF files.

        """
        if isinstance(path, str):
            path = [path]
        for file_path in path:
            for root, _, files in os.walk(file_path, followlinks=True):
                for filename in files:
                    if not filename.endswith(".nc"):
                        continue
                    filepath = os.path.join(root, filename)
                    with xr.open_dataset(filepath) as dset:
                        for key in dset.variables.keys():
                            if key not in self.variables:
                                self.variables[key] = []
                            self.variables[key].append(filepath)
        return self

    def _by_regex(self, group_regex: str) -> dict:
        """Create a partition of the variables by regex"""
        groups = {}
        for var, files in self.variables.items():
            to_remove = []
            for filename in files:
                match = re.search(group_regex, filename)
                if match:
                    grp = match.group(1)
                    if grp not in groups:
                        groups[grp] = {}
                    if var not in groups[grp]:
                        groups[grp][var] = []
                    groups[grp][var].append(filename)
                    to_remove.append(filename)
            for rem in to_remove:
                self.variables[var].remove(rem)
        return groups

    def _by_attr(self, group_attr: str) -> dict:
        """Create a partition of the variables by attr"""
        groups = {}
        for var, files in self.variables.items():
            to_remove = []
            for filename in files:
                with xr.open_dataset(filename) as dset:
                    if group_attr in dset.attrs:
                        grp = dset.attrs[group_attr]
                        if grp not in groups:
                            groups[grp] = {}
                        if var not in groups[grp]:
                            groups[grp][var] = []
                        groups[grp][var].append(filename)
                        to_remove.append(filename)
            for rem in to_remove:
                self.variables[var].remove(rem)
        return groups

    def create_submodels(
        self, child_regex: Union[str, None] = None, child_attr: Union[str, None] = None
    ) -> Self:
        """Create submodels automatically using regular expressions or globl attributes.

        Parameters
        ----------
        child_regex
            The regular expression from which to create subgroups where the label will
            be matches on the first defined group.
        child_attr
            The global attribute to use to create subgroups.

        """
        groups = {}
        if not groups and child_regex is not None:
            groups = self._by_regex(child_regex)
        if not groups and child_attr is not None:
            groups = self._by_attr(child_attr)
        for grp, submodel in groups.items():
            mod = ModelResult(name=grp)
            mod.variables = submodel
            self.children[mod.name] = mod
        return self

    def _get_variable_parent(self, vname: str) -> xr.Dataset:
        """Return the variable of the parent model and associate measures.

        This function is meant to be used internally. See `get_variable`.

        Parameters
        ----------
        vname
            The name of the variable to retrieve.

        """
        name = self.synonyms[vname] if vname in self.synonyms else vname
        if name not in self.variables:
            raise VarNotInModel(name, self)
        ds = xr.open_mfdataset(sorted(self.variables[name]))
        ds = ds.rename({name: vname})
        ds.attrs["varname"] = vname
        da = ds[vname]

        # add cell measures if appropriate
        if "cell_measures" not in da.attrs or (
            self.area_atm is None and self.area_ocn is None
        ):
            return ds
        if "areacella" in da.attrs["cell_measures"]:
            measures = self.area_atm
        if "areacello" in da.attrs["cell_measures"]:
            measures = self.area_ocn
        if "cell_methods" in da.attrs and "where land" in da.attrs["cell_methods"]:
            measures *= self.frac_lnd
        ds, measures = xr.align(ds, measures, join="override")
        ds["cell_measures"] = measures
        return ds

    def get_variable(self, vname: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        """Return the variable and associate measures.

        If the model has children defined, then return a dictionary whose keys are the
        children models and the entries are the variable for each. If no children are
        defined, return the variable of the parent.

        Parameters
        ----------
        vname
            The name of the variable to retrieve.

        """
        if not self.children:
            return self._get_variable_parent(vname)
        var = {}
        for name, child in self.children.items():
            var[name] = child.get_variable(vname)
        return var

    def setup_grid_information(self):
        """Associate cell measures and land fractions with this model."""
        atm = ocn = lnd = None
        try:
            atm = self._get_variable_parent("areacella")["areacella"]
        except VarNotInModel:
            pass
        try:
            ocn = self._get_variable_parent("areacello")["areacello"]
        except VarNotInModel:
            pass
        try:
            lnd = self._get_variable_parent("sftlf")["sftlf"]
            if np.allclose(lnd.max(), 100):
                lnd *= 0.01
        except VarNotInModel:
            pass
        if atm is not None and lnd is not None:
            atm, lnd = xr.align(atm, lnd, join="override", copy=False)
        if atm is not None:
            self.area_atm = atm
        if ocn is not None:
            self.area_ocn = ocn
        if lnd is not None:
            self.frac_lnd = lnd

    def add_child(self, mod: Union[Self, list[Self]]) -> None:
        """Add a child model to this model."""
        if isinstance(mod, ModelResult):
            mod = [mod]
        for child in mod:
            if child.name not in self.children:
                self.children[child.name] = child

    def add_synonym(self, this_name: str, known_as: str) -> None:
        """Add variable synonyms to the model.

        Once associated with the model, `get_variable` will automatically check and
        return `this_name` when the variable `known_as` is requested.

        Parameters
        ----------
        this_name
            The name of the variable associated with this model.
        known_as
            The name that the variable will also be known as.

        """
        assert this_name in self.variables
        self.synonyms[known_as] = this_name

    def to_pickle(self, filename: str) -> None:
        """Save this model object as a pickle file."""
        with open(filename, mode="wb") as pkl:
            pickle.dump(self.__dict__, pkl)

    def read_pickle(self, filename: str) -> Self:
        """Create a model object from a pickle file.

        Examples
        --------
        > m = ModelResult().read_pickle("model.pkl")

        """
        with open(filename, mode="rb") as pkl:
            obj = self.__new__(self.__class__)
            obj.__dict__.update(pickle.load(pkl))
        return obj

    def read_yaml(self, filename: str):
        """Create a model object from a yaml file.

        Examples
        --------
        > m = ModelResult().read_yaml("model.yaml")

        """

        def _read_single_entry(name, opts):
            m = ModelResult(name=name)
            if "children" in opts:
                for child in opts["children"]:
                    m.add_child(_read_single_entry(child, opts["children"][child]))
            elif "paths" in opts:
                m.find_files(opts["paths"])
            if "synonyms" in opts:
                for from_name, to_name in opts["synonyms"].items():
                    m.add_synonym(from_name, to_name)
            m.setup_grid_information()
            return m

        models = []
        with open(filename, encoding="utf-8") as fin:
            yml = yaml.safe_load(fin)
            for name, opts in yml.items():
                models.append(_read_single_entry(name, opts))
        if len(models) == 1:
            return models[0]
        return models


class VarNotInModel(Exception):
    """A exception to indicate that a variable is not present in the model
    results."""

    def __init__(self, variable: str, model: ModelResult):
        super().__init__(f"{variable} not found in {model.name}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = ModelResult().read_yaml("model_group.yaml")
    print(m)
    v = m.get_variable("gpp")

    fig, axs = plt.subplots(nrows=len(v), tight_layout=True)
    for i, (name, var) in enumerate(v.items()):
        var["gpp"].mean(dim="time").plot(ax=axs[i])
        axs[i].set_title(name)
    plt.show()
