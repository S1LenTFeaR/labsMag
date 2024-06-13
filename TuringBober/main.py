import numpy as np
import matplotlib.pyplot as plt

class Turing:
    def __init__(self, tape_input, head_input):
        self.tape = [] # лента
        for i in range(0, len(tape_input)):
            self.tape.append(int(tape_input[i]))
        self.head = head_input  # положение головки
        self.state = "A"  # состояние
        # массивы для вывода на экран
        self.tape_render = np.array(self.tape)
        self.head_render = [self.head, 0]
        self.state_render = self.state

    def move(self, direction):
        if direction == "L":
            self.head -= 1
        elif direction == "R":
            self.head += 1

    def transition(self):
        current_symbol = self.tape[self.head]

        # Правила для усердного бобра с тремя состояниями
        if self.state == "A":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
            elif current_symbol == 1:
                self.tape[self.head] = 1
                self.move("L")
                self.state = "C"
        elif self.state == "B":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("L")
                self.state = "A"
            elif current_symbol == 1:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
        elif self.state == "C":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
            elif current_symbol == 1:
                self.tape[self.head] = 0
                self.move("L")
                self.state = "C"
    def go(self, steps):
        i = 0
        # пока счетчик меньше максимального количества шагов
        while i < steps:
            # рабочее пространство бобра от точки старта до начала холста
            working_space = np.array(self.tape)[:head + 1]
            # если не все рабочее пространство под бобром закрашено
            if not np.all(working_space == 1):
                # перемещение головки
                self.transition()
                # добавление нового состояния ленты
                self.tape_render = np.vstack((self.tape_render, self.tape))
                # добавление нового положения головки
                if (self.head < 0):
                    self.head_render = np.vstack((self.head_render, [self.head + len(self.tape), i + 1]))
                else:
                    self.head_render = np.vstack((self.head_render, [self.head, i + 1]))
                # добавление нового состояния головки
                self.state_render += self.state
                i += 1
            else:
                break
        # отрисовка результата
        self.draw()
    def draw(self):
        plt.figure(figsize=(8, 8))
        plt.imshow(self.tape_render, cmap='Oranges', interpolation='none')
        plt.scatter(self.head_render[:, 0], self.head_render[:, 1], color='black', marker='.')
        i = 0
        for s in self.state_render:
            plt.annotate(s, (self.head_render[i, 0], self.head_render[i, 1]), color='red', weight='bold')
            i += 1
        plt.show()

class Turing:
    def __init__(self, tape_input, head_input):
        self.tape = [] # лента
        for i in range(0, len(tape_input)):
            self.tape.append(int(tape_input[i]))
        self.head = head_input  # положение головки
        self.state = "A"  # состояние
        # массивы для вывода на экран
        self.tape_render = np.array(self.tape)
        self.head_render = [self.head, 0]
        self.state_render = self.state

    def move(self, direction):
        if direction == "L":
            self.head -= 1
        elif direction == "R":
            self.head += 1

    def transition(self):
        current_symbol = self.tape[self.head]

        # Правила для усердного бобра с тремя состояниями
        if self.state == "A":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
            elif current_symbol == 1:
                self.tape[self.head] = 1
                self.move("L")
                self.state = "C"
        elif self.state == "B":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("L")
                self.state = "A"
            elif current_symbol == 1:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
        elif self.state == "C":
            if current_symbol == 0:
                self.tape[self.head] = 1
                self.move("R")
                self.state = "B"
            elif current_symbol == 1:
                self.tape[self.head] = 0
                self.move("L")
                self.state = "C"
    def go(self, steps):
        i = 0
        # пока счетчик меньше максимального количества шагов
        while i < steps:
            # рабочее пространство бобра от точки старта до начала холста
            working_space = np.array(self.tape)[:head + 1]
            # если не все рабочее пространство под бобром закрашено
            if not np.all(working_space == 1):
                # перемещение головки
                self.transition()
                # добавление нового состояния ленты
                self.tape_render = np.vstack((self.tape_render, self.tape))
                # добавление нового положения головки
                if (self.head < 0):
                    self.head_render = np.vstack((self.head_render, [self.head + len(self.tape), i + 1]))
                else:
                    self.head_render = np.vstack((self.head_render, [self.head, i + 1]))
                # добавление нового состояния головки
                self.state_render += self.state
                i += 1
            else:
                break
        # отрисовка результата
        self.draw()
    def draw(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.tape_render, cmap='Blues', interpolation='none')
        plt.scatter(self.head_render[:, 0], self.head_render[:, 1], color='black', marker='o')
        i = 0
        for s in self.state_render:
            plt.annotate(s, (self.head_render[i, 0], self.head_render[i, 1]), color='red', weight='bold')
            i += 1
        plt.show()

# начальное состояние ленты
tape = "000000"
# начальное положение головки
head = 2
# создание бобра
turing = Turing(tape, head)
# максимальное количество шагов
steps = 50
# запуск копания бобра
turing.go(steps)
