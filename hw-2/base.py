import exceptions


class Vechile():
    def __init__(self, weight, fuel, fuel_consumptiong):
        self.weight = weight  # tons
        self.started = False
        self.fuel = fuel  # liters
        self.fuel_consumption = fuel_consumptiong  # liters per hour

    def start(self):
        if self.fuel > 0:
            self.started = True
        else:
            raise exceptions.LowFuelError

    def move(self, distance):
        if (self.fuel / self.fuel_consumption) >= distance:
            pass
        else:
            raise exceptions.NotEnoughFuel(distance)
