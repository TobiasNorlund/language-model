from absl import flags


def get(name):
    """
    Just use absl as global backend to hold actual values
    """
    return flags.FLAGS[name].value


def add(name, default_value=None, dtype=None, **kwargs):
    """
    Adds a global hyperparameter

    A hyperparameter will get an absl flag defined, and will be required unless a default value is provided

    :param name:
    :param default_value:
    :param dtype:
    :param kwargs:
    :return:
    """
    dtype = dtype if dtype is not None else type(default_value)
    if dtype == int:
        flags.DEFINE_integer(name, default_value, **kwargs)
    elif dtype == str:
        flags.DEFINE_string(name, default_value, **kwargs)
    elif dtype == float:
        flags.DEFINE_float(name, default_value, **kwargs)
    elif dtype == list:
        flags.DEFINE_enum(name, default_value, **kwargs)
    elif dtype == bool:
        flags.DEFINE_bool(name, default_value, **kwargs)
    else:
        raise RuntimeError("Unsupported type for hparam {}".format(name))
    if default_value is None:
        flags.mark_flag_as_required(name)