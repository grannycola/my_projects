import exceptions


class Vehicle:
    def __init__(self, weight, fuel, fuel_consumption):
        self.weight = weight
        self.started = False
        self.fuel = fuel
        self.fuel_consumption = fuel_consumption

    def start(self):
        if self.fuel > 0:
            self.started = True
        else:
            raise exceptions.LowFuelError

    def move(self, distance):
        if (self.fuel_consumption * distance) <= self.fuel:
            self.fuel -= self.fuel_consumption * distance
        else:
            raise exceptions.NotEnoughFuel(distance)
