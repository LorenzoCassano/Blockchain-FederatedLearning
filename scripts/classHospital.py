class Hospital:
    def __init__(self, name,dataset_name):
        self._dataset_name = dataset_name
        self._name = name
        self._address = None
        self._model = None
        self._compile_info = None
        self._aggregated_weights = None
        self._weights = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def address(self):
        return self._address

    @address.setter
    def address(self, value):
        self._address = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def compile_info(self):
        return self._compile_info

    @compile_info.setter
    def compile_info(self, value):
        self._compile_info = value

    @property
    def aggregated_weights(self):
        return self._aggregated_weights

    @aggregated_weights.setter
    def aggregated_weights(self, value):
        self._aggregated_weights = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    def __str__(self):
        return f"{self._dataset_name}"
