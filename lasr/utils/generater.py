import importlib
import inspect
import logging

def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)



class BaseConfig(object):
    def __init__(self, name, kwargs={}):
        self.conf_dict = kwargs
        self.name = name
        if name == None or name == "None":
            self.name = None
            return 
        self.conf_class = None
        if not self.config_initial():
            raise RuntimeError("config is wrong")

    def config_initial(self):
        self.conf_class = dynamic_import(self.name)
        self.check_kwargs(self.conf_class, self.conf_dict)
        return True

    def generateExample(self, *args, **kwargs):
        if self.name == None:
            return None
        # FIXME
        new_kwargs = {}
        new_kwargs.update(self.conf_dict)
        new_kwargs.update(kwargs)
        return self.conf_class(*args, **new_kwargs)

    def __getitem__(self, index):
        return self.conf_dict[index]

    def __setitem__(self, index, value):
        if index in self.conf_dict:
            self.conf_dict[index] = value
        else:
            raise RuntimeWarning(index + 'is not in this config')

    def get_conf_dict(self):
        return self.conf_dict


    @staticmethod
    def check_kwargs(cls, kwargs, name=None):
        '''
            使用inspect检测kwargs和cls是否兼容
            如果kwargs中存在额外的参数，raise error
            如果kwargs缺少必须的参数，raise error，并给出正确的dict的格式
            如果kwargs缺少非必须的参数，补充为默认值，raise warning
        '''
        def _default_params(cls):
            try:
                d = dict(inspect.signature(cls.__init__).parameters)
            except ValueError:
                d = dict()
            return {
                k: v.default for k, v in d.items() if v.default != inspect.Parameter.empty and k != "self"
            }
        def _required_params(cls):
            try:
                d = dict(inspect.signature(cls.__init__).parameters)
            except ValueError:
                d = dict()
            return {
                k: v.default for k, v in d.items() if v.default == inspect.Parameter.empty and k != "self"
            }
            
        try:
            params = inspect.signature(cls.__init__).parameters
        except ValueError:
            return
        if name is None:
            name = cls.__name__
        for k in kwargs.keys():
            if k not in params:
                raise ValueError("initialization of class '{0}' got an unexpected keyword argument '{1}', "
                                 "the standard config should be {2}".format(name, k, params))
        # for k in _required_params(cls):
        #     if k not in kwargs:
        #         raise UserWarning("initialization of class '{0}' require keyword argument '{1}', "
        #                          "the standard config should be {2}".format(name, k, params))
        # for k in _default_params(cls):
        #     if k not in kwargs.keys():
        #         # logging.warning("initialization of class '{0}' require keyword argument '{1}', "
        #         #               "set to default value {2}".format(name, k, _default_params(cls)[k]))
        #         raise UserWarning("initialization of class '{0}' require keyword argument '{1}', "
        #                       "set to default value {2}".format(name, k, _default_params(cls)[k]))
