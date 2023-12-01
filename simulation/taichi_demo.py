import taichi as ti


ti.init()


def f(x: float) -> float:
    return x + 1.


wf = ti.func(f)


@ti.kernel
def demo(x: float) -> float:
    return wf(x)


if __name__ == '__main__':
    print(demo(0.))
    print(f(0.))
