from base import Vechile


class Car(Vechile):
    def __init__(self, weight, fuel, fuel_consumptiong):
        super().__init__(weight, fuel, fuel_consumptiong)
        self.engine = None

    def set_engine(self, engine_cls):
        self.engine = engine_cls

