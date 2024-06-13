 
import asyncio
from IPython.display import display_html, clear_output, display

#Модель горящего леса
class ForestFireModel(Cell2D):
    def __init__(self, p, f, size):
        self.p = p
        self.f = f
        self.size = size
        self.grid = np.zeros((size, size))

    def initialize(self):
        self.grid = np.random.choice([0, 1], size=(self.size, self.size), p=[1 - self.p, self.p])

    def step(self):
        new_grid = np.zeros((self.size, self.size))

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    new_grid[i][j] = np.random.choice([0, 1], p=[1 - self.p, self.p])
                elif self.grid[i][j] == 1:
                    neighbors = self.get_neighbors(i, j)
                    if any(self.grid[x][y] == 2 for x, y in neighbors):
                        new_grid[i][j] = 2
                    else:
                        new_grid[i][j] = np.random.choice([1, 2], p=[1 - self.f, self.f])
                elif self.grid[i][j] == 2:
                    new_grid[i][j] = 0

        self.grid = new_grid

    def get_neighbors(self, i, j):
        neighbors = []

        for x in range(max(0, i - 1), min(i + 2, self.size)):
            for y in range(max(0, j - 1), min(j + 2, self.size)):
                if (x, y) != (i, j):
                    neighbors.append((x, y))

        return neighbors

    def simulate(self, max_step = None, isPrint = True):
        self.initialize()
        step = 0
        if isPrint:
            plt.figure()

        while True:
            if isPrint:
                plt.imshow(self.grid, cmap='hot', vmin=0, vmax=2)
                plt.title('Forest Fire Model')
                plt.text(46, -2, f"steps: {step}")
                plt.pause(0.01)
                clear_output(wait=True)
                plt.clf()

            prev_grid = np.copy(self.grid)
            self.step()
            step += 1
            if step == max_step:
                if isPrint != True:
                    plt.figure()
                    plt.imshow(self.grid, cmap='hot', vmin=0, vmax=2)
                    plt.title('Forest Fire Model')
                    plt.show()
                return self.grid
            if np.array_equal(prev_grid, self.grid):
                break

        if isPrint:
            plt.imshow(self.grid, cmap='hot', vmin=0, vmax=2)
            plt.title('Stable State')
            plt.show()

# Задаем параметры модели
p = 0.01
f = 0.001
# Размер матрицы
size = 50
# Количество шагов по времени
max_step = 100
np.random.seed(22)
# Создаем и запускаем модель с
model = ForestFireModel(p, f, size)
array = model.simulate(max_step, isPrint = True)

# Проверка на фрактальность
def plot_perc_scaling(sizes, max_step, q):
    res = []
    resf = []
    for size in sizes:
        model = ForestFireModel(p, f, size)
        array = np.array(model.simulate(max_step, isPrint = False))
        if array.any():
            trees = np.sum(array == 1)
            treesFire = np.sum(array == 2)
            num_trees = trees
            num_treesFire = treesFire
            res.append((size, size**2, num_trees))
            resf.append((size, size ** 2, num_treesFire))
    # Количество ячеек в перколяционном кластере в сравнении с размером клеточного автомата
    sizes, cells, filled = zip(*res)
    options = dict(linestyle='dashed', color='gray', alpha=0.7)
    plt.plot(sizes, cells, label='d=2', **options)
    plt.plot(sizes, filled, '.', label='filled')
    plt.plot(sizes, sizes, label='d=1', **options)
    decorate(xlabel='Array Size',
             ylabel='Cell Count',
             xscale='log', xlim=[9, 110],
             yscale='log', ylim=[9, 20000],
             loc='upper left')
    plt.show()
    # количество клеток в каждом перколяционном кластере
    for ys in [cells, filled, sizes]:
        params = linregress(np.log(sizes), np.log(ys))
        print(params[0])

# Размерность от 30 до 50
sizes = np.arange(30, 51)
# При 65 временных шагах
plot_perc_scaling(sizes, 65, q=0.59)
