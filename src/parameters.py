# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under a Microsoft Research License.

import numpy as np
from distributions import TfKumaraswamy, TfLogNormal, TfNormal, TfTruncatedNormal, TfConstant
from utils import default_get_value

class DistributionDescription(object):
    def __init__(self, name, class_type, defaults):
        self.name = name
        self.class_type = class_type
        self.defaults = defaults
        self.other_params = {}

        if class_type == TfNormal or  class_type == TfLogNormal:
            self.nbr_free_params = 2
            self.free_params = ["mu", "log_prec"]
            self.params = ["mu", "prec"]
            self.free_to_constrained = ["identity", "positive"]

            mu_dependency = None
            prec_dependency = None

            init_mu = 0.0
            if "mu" in defaults:
                if defaults["prec"].__class__ == str:
                    print("found dependency for %s.mu = %s" % (self.name, defaults["mu"]))
                    mu_dependency = defaults["mu"]
                else:
                    init_mu = defaults["mu"]

            init_prec = 1.0
            init_log_prec = 0.0
            if "prec" in defaults:
                if (defaults["prec"] is not None) and (defaults["prec"].__class__ != str):
                    init_prec = defaults["prec"]
                    init_log_prec = np.log(init_prec)
                elif defaults["prec"].__class__ == str:
                    print("found dependency for %s.prec = %s" % (self.name, defaults["prec"]))
                    prec_dependency = defaults["prec"]
            elif "sigma" in defaults:
                if (defaults["sigma"] is not None) and (defaults["sigma"].__class__ != str):
                    init_prec = 1.0 / np.square(defaults["sigma"])
                    init_log_prec = np.log(init_prec)
            else:
                print("WARNING: using *very* default params for %s" % (str(class_type)))

            self.init_params = [init_mu, init_prec]
            self.init_free_params = [init_mu, init_log_prec]
            self.dependencies = [mu_dependency, prec_dependency]

        elif class_type == TfTruncatedNormal:
            self.nbr_free_params = 2
            self.free_params = ["mu", "log_prec"]
            self.params = ["mu", "prec"]
            self.free_to_constrained = ["identity", "positive"]
            self.other_param_names = ["a", "b"]

            mu_dependency = None
            prec_dependency = None

            init_mu = 0.0
            if "mu" in defaults:
                if defaults["prec"].__class__ == str:
                    print("found dependency for %s.mu = %s" % (self.name, defaults["mu"]))
                    mu_dependency = defaults["mu"]
                else:
                    init_mu = defaults["mu"]

            init_prec = 1.0
            init_log_prec = 0.0
            if "prec" in defaults:
                if (defaults["prec"] is not None) and (defaults["prec"].__class__ != str):
                    init_prec = defaults["prec"]
                    init_log_prec = np.log(init_prec)
                elif defaults["prec"].__class__ == str:
                    print("found dependency for %s.prec = %s" % (self.name, defaults["prec"]))
                    prec_dependency = defaults["prec"]
            elif defaults.has_key("sigma"):
                if (defaults["sigma"] is not None) and (defaults["sigma"].__class__ != str):
                    init_prec = 1.0 / np.square(defaults["sigma"])
                    init_log_prec = np.log(init_prec)
            else:
                print("WARNING: using *very* default params for %s" % (str(class_type)))

            self.init_params = [init_mu, init_prec]
            self.init_free_params = [init_mu, init_log_prec]
            self.dependencies = [mu_dependency, prec_dependency]

            self.other_params["a"] = default_get_value(defaults, "a", -np.inf)
            self.other_params["b"] = default_get_value(defaults, "b", -np.inf)

        elif class_type == TfKumaraswamy:
            self.nbr_free_params = 2
            self.free_params = ["log_a", "log_b"]
            self.params = ["a", "b"]
            self.free_to_constrained = ["positive", "positive"]
            self.other_param_names = ["zmin", "zmax"]

            #init_zmin = 0.0
            #init_zmax = 1.0

            a_dependency = None
            b_dependency = None

            init_a = None
            init_log_a = None
            if "a" in defaults:
                if (defaults["a"] is not None) and (defaults["a"].__class__ != str):
                    init_a = defaults["a"]
                    init_log_a = np.log(init_a)
                elif defaults["a"].__class__ == str:
                    print("found dependency for %s.a = %s" % (self.name, defaults["a"]))
                    b_dependency = defaults["a"]
            else:
                raise NotImplementedError("Missing a")

            init_b = None
            init_log_b = None
            if "b" in defaults:
                if (defaults["b"] is not None) and (defaults["a"].__class__ != str):
                    init_b = defaults["b"]
                    init_log_b = np.log(init_b)
                elif defaults["b"].__class__ == str:
                    print("found dependency for %s.b = %s" % (self.name, defaults["b"]))
                    b_dependency = defaults["b"]
            else:
                raise NotImplementedError("Missing b")

            self.init_params = [init_a, init_b]
            self.init_free_params = [init_log_a, init_log_b]
            self.dependencies = [a_dependency, b_dependency]

            self.other_params["zmin"] = default_get_value(defaults, "zmin", 0.0)
            self.other_params["zmax"] = default_get_value(defaults, "zmax", 1.0)
        elif class_type == TfConstant:
            self.nbr_free_params = 1
            self.free_params = ["value"]
            self.params = ["value"]
            self.free_to_constrained = ["identity"]

            value_dependency = None
            
            init_value = 0.0
            if "value" in defaults:
                init_value = defaults["value"]

            self.init_params = [init_value]
            self.init_free_params = [init_value]
            self.dependencies = [value_dependency]
        else:
            raise NotImplementedError("unknown class type %s" % (str(class_type)))

    def __str__(self):
        s = "%s " % (str(self.class_type))
        for k, v in self.defaults.items():
            s += "%s=%s  " % (k, str(v))

        for k, dependency in zip(self.params, self.dependencies):
            if dependency is not None:
                s += "%s DEPENDENCE on %s  " % (k, dependency)

        return s

def instantiate_from_specs(name, specs, conditioning):
    try:  # catch conditions dict and skip
        sd = specs['distribution']
    except:
        print("instantiate_from_specs:: skip instantiate")
        return None
    if sd == "Normal":
        mu = default_get_value(specs, "mu", 0.0)
        sigma = default_get_value(specs, "sigma", None)
        prec = default_get_value(specs, "prec", None)
        s = {'mu':mu, "sigma":sigma, "prec":prec, "c":conditioning}
        return DistributionDescription(name, TfNormal, s) #TfNormal(**s)
    if sd == "LogNormal":
        mu = default_get_value(specs, "mu", 0.0)
        sigma = default_get_value(specs, "sigma", None)
        prec = default_get_value(specs, "prec", None)
        s = {'mu':mu, "sigma":sigma, "prec":prec, "c":conditioning}
        return DistributionDescription(name, TfLogNormal, s) #TfLogNormal(**s) #mu=float(specs['mu']), scale=float(specs['scale']))
    if sd == "TruncNormal":
        mu = default_get_value(specs, "mu", 0.0)
        sigma = default_get_value(specs, "sigma", None)
        prec = default_get_value(specs, "prec", None)
        a = default_get_value(specs, "a", -np.inf)
        b = default_get_value(specs, "b", np.inf)
        s = {'mu':mu, "sigma":sigma, "prec":prec, "a":a, "b":b, "c":conditioning}
        return DistributionDescription(name, TfTruncatedNormal, s) #TfLogNormal(**s) #mu=float(specs['mu']), scale=float(specs['scale']))
    if sd == "Kumaraswamy":
        a = default_get_value(specs, "a", None)
        b = default_get_value(specs, "b", None)
        zmin = default_get_value(specs, "zmin", 0.0)
        zmax = default_get_value(specs, "zmax", 1.0)
        s = {'a':a, "b":b, "zmin":zmin, "zmax":zmax, "c":conditioning}
        return DistributionDescription(name, TfKumaraswamy, s) #TfLogNormal(**s) #mu=float(specs['mu']), scale=float(specs['scale']))
    if sd == "Constant":
        value = default_get_value(specs, "value", 0.0)
        s = {'value':value}
        return DistributionDescription(name, TfConstant, s)
    print("instantiate_from_specs:: cannot instantiate %s" % sd)
    return None


class DotOperatorParams(object):
    def __init__(self):
        self.list_of_params = []

    def get_parameter_counts(self):
        return len(self.list_of_params)

    def add_from_existing(self, _name_of_param, _p_value):
        raise Exception("add_from_existing: now always from specs")

    def add_from_spec(self, name_of_param, spec_of_param, conditioning):
        if hasattr(self, name_of_param):
            print("already have param named: ", name_of_param)
        else:
            setattr(self, name_of_param, instantiate_from_specs(name_of_param, spec_of_param, conditioning))
            self.list_of_params.append(name_of_param)

    def __str__(self):
        lines = ["%s = %s\n" % (p_name, getattr(self, p_name)) for p_name in self.list_of_params]
        return ''.join(lines)

class Parameters(object):
    def __init__(self):
        pass

    def get_parameter_counts(self):
        n_local = self.l.get_parameter_counts() if hasattr(self, "l") else 0
        n_global_cond = self.g_c.get_parameter_counts() if hasattr(self, "g_c") else 0
        n_global = self.g.get_parameter_counts() if hasattr(self, "g") else 0
        n_constant = self.c.get_parameter_counts() if hasattr(self, "c") else 0
        return n_local, n_global_cond, n_global, n_constant

    def add_shared(self, p):
        self.s = p

    def add_global(self, p):
        self.g = p

    def add_global_cond(self, p):
        self.g_c = p

    def add_local(self, p):
        self.l = p

    def add_constant(self, p):
        self.c = p

    def is_shared(self, p_name):
        return hasattr(self, "s") and hasattr(self.s, p_name)

    def is_global(self, p_name):
        return hasattr(self, "g") and hasattr(self.g, p_name)

    def is_global_cond(self, p_name):
        return hasattr(self, "g_c") and hasattr(self.g_c, p_name)

    def is_local(self, p_name):
        return hasattr(self, "l") and hasattr(self.l, p_name)

    def is_constant(self, p_name):
        return hasattr(self, "c") and hasattr(self.c, p_name)

    def get_shared(self, p_name):
        if self.is_shared(p_name):
            return getattr(self.s, p_name)
        raise AttributeError

    def get_global(self, p_name):
        if self.is_global(p_name):
            return getattr(self.g, p_name)
        raise AttributeError

    def get_global_cond(self, p_name):
        if self.is_global_cond(p_name):
            return getattr(self.g_c, p_name)
        raise AttributeError

    def get_local(self, p_name):
        if self.is_local(p_name):
            return getattr(self.l, p_name)
        raise AttributeError

    def get_constant(self, p_name):
        if self.is_constant(p_name):
            return getattr(self.c, p_name)
        raise AttributeError


    def pretty_print(self):
        if hasattr(self, "s"):
            print("-----------------\nSHARED parameters\n-----------------")
            print(self.s)
        if hasattr(self, "g"):
            print("-----------------\nGLOBAL parameters\n-----------------")
            print(self.g)
        if hasattr(self, "g_c"):
            print("-----------------\nGLOBAL conditioned parameters\n-----------------")
            print(self.g_c)
        if hasattr(self, "l"):
            print("-----------------\nLOCAL parameters\n-----------------")
            print(self.l)
        if hasattr(self, "c"):
            print("-----------------\nCONSTANT parameters\n-----------------")
            print(self.c)

    def load(self, params_dict):
        # Store the incoming "params_dict" object (originating from the YAML file) to save
        # having to pass around both a Parameters object and the params in various places.
        self.params_dict = params_dict
        self.load_shared_distributions(params_dict)
        self.load_global(params_dict)
        self.load_global_cond(params_dict)
        self.load_local(params_dict)
        self.load_constant(params_dict)

    def load_for_flow(self, params_dict):
        #self.load_shared_distributions(params_dict)
        self.load_global(params_dict, keyword="global_flow")
        self.load_local(params_dict, keyword="local_flow")

    def load_shared_distributions(self, params_dict):
        keyword = 'shared'
        p = DotOperatorParams()

        if keyword not in params_dict:
            print("load_%s_params:: None found in params_dict" % keyword)
            return

        keyword_dict = params_dict[keyword]
        conditioning = None

        for k, v in keyword_dict.items():
            if k == 'conditioning':
                raise Exception("shared_distributions can no longer have conditioning")
            p.add_from_spec(k, v, conditioning)

        self.add_shared(p)

    def load_constant(self, params_dict, keyword="constant"):

        p = DotOperatorParams()
        if keyword not in params_dict:
            print("load_%s_params:: None found in params_dict" % keyword)
            return

        keyword_dict = params_dict[keyword]
        # get conditioning statements first
        conditioning = None
        if 'conditioning' in keyword_dict:
            raise Exception("constant params can't have conditioning") #

        for k, v in keyword_dict.items():
            if k == 'conditioning':
                continue
            p.add_from_spec(k, {'distribution':'Constant', 'value':v}, conditioning)

        self.add_constant(p)

    def load_global(self, params_dict, keyword="global"):

        p = DotOperatorParams()
        if keyword not in params_dict:
            print("load_%s_params:: None found in params_dict" % keyword)
            return

        keyword_dict = params_dict[keyword]
        # get conditioning statements first
        conditioning = None
        if 'conditioning' in keyword_dict:
            raise Exception("global_params can no longer have conditioning") #

        for k, v in keyword_dict.items():
            if k == 'conditioning':
                continue
            if self.is_shared(v['distribution']):
                # get the spec info for shared
                v = params_dict['shared'][v['distribution']]
            p.add_from_spec(k, v, conditioning)

        self.add_global(p)

    def load_global_cond(self, params_dict, keyword="global_conditioned"):

        #keyword = 'global'
        p = DotOperatorParams()

        if keyword not in params_dict:
            print("load_%s_params:: None found in params_dict" % keyword)
            return

        keyword_dict = params_dict[keyword]

        # get conditioning statements first
        conditioning = None
        if 'conditioning' in keyword_dict:
            conditioning = keyword_dict['conditioning']
            if 'species' in conditioning:
                assert conditioning['species'] is False, "cannot have species here"
        else:
            raise Exception("global_cond MUST have conditioning") #

        for k, v in keyword_dict.items():
            if k == 'conditioning':
                continue
                #conditioning = v  # capture conditions for these parameters
            else:
                if self.is_shared(v['distribution']):
                    p.add_from_spec(k, params_dict['shared'][v['distribution']], conditioning)
                    #p.add_from_existing(k, self.get_shared(v['distribution']))
                else:
                    p.add_from_spec(k, v, conditioning)

        self.add_global_cond(p)

    def load_local(self, params_dict, keyword="local"):
        #keyword = 'local'
        p = DotOperatorParams()

        if keyword not in params_dict:
            print("load_%s_params:: None found in params_dict" % keyword)
            return

        keyword_dict = params_dict[keyword]
        # get conditioning statements first
        conditioning = None
        if 'conditioning' in keyword_dict:
            conditioning = keyword_dict['conditioning']

        for k, v in keyword_dict.items():
            if k == 'conditioning':
                pass
            elif self.is_shared(v['distribution']):
                p.add_from_spec(k, params_dict['shared'][v['distribution']], conditioning)
            elif self.is_global(v['distribution']) or self.is_global_cond(v['distribution']):
                raise Exception("locals can only inherit from shared")
            else:
                p.add_from_spec(k, v, conditioning)
        self.add_local(p)