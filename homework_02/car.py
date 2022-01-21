from homework_02.base import Vehicle


class Car(Vehicle):
    def __init__(self, weight, fuel, fuel_consumption):
        super().__init__(weight, fuel, fuel_consumption)
        self.engine = None

    def set_engine(self, engine_cls):
        self.engine = engine_cls
