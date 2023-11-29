import taichi as ti


ti.init()


@ti.data_oriented
class Sampler:
    @ti.func
    def sample(self) -> float:
        return 0.


@ti.data_oriented
class Demo:
    def __init__(self):
        self.use_sampler = False
        self.sampler = Sampler()
        self.fixed_number = 1.

    @ti.kernel
    def run(self) -> float:
        res = 0.
        if self.use_sampler:
            res = self.sampler.sample()
        else:
            res = self.fixed_number
        return res


if __name__ == '__main__':
    d = Demo()
    print(d.run())
