from absl import flags


class HParamSet:
    """
    Wraps the absl flags to static object attributes of this class
    """

    def add(self, name, default_value, t=None, required=False, **kwargs):
        t = t if t is not None else type(default_value)
        if t == int:
            flags.DEFINE_integer(name, default_value, **kwargs)
        elif t == str:
            flags.DEFINE_string(name, default_value, **kwargs)
        elif t == float:
            flags.DEFINE_float(name, default_value, **kwargs)
        else:
            raise RuntimeError("Unsupported type for hparam {}".format(name))
        if required is True:
            flags.mark_flag_as_required(name)

    def __getattr__(self, item):
        return flags.FLAGS[item].value
