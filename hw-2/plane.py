from base import Vechile
import exceptions


class Plane(Vechile):
    def __init__(self, weight, fuel, fuel_consumption, cargo, max_cargo):
        super().__init__(weight, fuel, fuel_consumption)
        self.cargo = cargo
        self.max_cargo = max_cargo

    def load_cargo(self, weight):
        if (weight + self.cargo) >= self.max_cargo:
            raise exceptions.CargoOverload(weight + self.cargo, self.max_cargo)
        else:
            self.cargo += weight + self.cargo

    def remove_all_cargo(self, last_cargo):
        self.cargo = 0
        return last_cargo
