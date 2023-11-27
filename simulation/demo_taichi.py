import taichi as ti
import numpy as np


def demo_parallel() -> None:
    @ti.kernel
    def run():
        for i in range(f.shape[0]):
            acc = 0
            for j in range(f.shape[1]):
                f[i, j] = acc
                acc += 1

    ti.init()
    f = ti.field(dtype=int, shape=(300_000, 1000))
    run()
    print(f)
    print(f.dtype)


def demo_numpy() -> None:
    @ti.kernel
    def run(x: ti.types.ndarray(dtype=int, ndim=1)) -> int:
        res = 0
        for i in x:
            res += x[i]
        return res

    ti.init()
    x = np.array([0, 2, 4])
    res = run(x)
    print(res)


@ti.data_oriented
class MemoryLeak:
    def run(self):
        self._create_fields()
        self._sim()

    def _create_fields(self):
        self.values = ti.field(dtype=float, shape=(300_000, 2_000))

    @ti.kernel
    def _sim(self):
        self.values[0, 0] = 100


def memory() -> None:
    ti.init()
    c = MemoryLeak()
    for i in range(10):
        print(f"Run {i}")
        c.run()


if __name__ == '__main__':
    memory()
