class LowFuelError(Exception):
    def __str__(self):
        return "Your vehicle is out of fuel to start"


class NotEnoughFuel(Exception):
    def __init__(self, distance):
        self.distance = distance

    def __str__(self):
        return f"Your vehicle does not have enough fuel to cover the specified distance({self.distance} km)"


class CargoOverload(Exception):
    def __init__(self, total_weight, max_cargo):
        self.total_weight = total_weight
        self.max_cargo = max_cargo

    def __str__(self):
        return f"Your vehicle does not have enough capacity (your capacity = {self.max_cargo} tons) " \
               f"to carry a load of {self.total_weight} tons"

