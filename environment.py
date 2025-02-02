import random
import torch
from utils import BLOCK_SIZE, GRID_WIDTH, GRID_HEIGHT, UP, RIGHT, DOWN, LEFT, is_danger, get_new_direction

class Snake:
    def __init__(self, init_pos, init_direction, color):
        self.positions = [init_pos]  # список координат (x, y)
        self.direction = init_direction
        self.color = color
        self.alive = True
        self.score = 0

    def move(self, new_direction, food_eaten):
        """
        Перемещает змейку: добавляет новую голову; если еда не съедена – удаляет хвост.
        """
        self.direction = new_direction
        head_x, head_y = self.positions[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.positions.insert(0, new_head)
        if not food_eaten:
            self.positions.pop()

class SnakeGame:
    def __init__(self, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT):
        self.grid_width = grid_width
        self.grid_height = grid_height
        # Инициализируем две змейки в разных частях поля
        self.snake1 = Snake(init_pos=(grid_width // 4, grid_height // 2), init_direction=RIGHT, color=(0, 255, 0))
        self.snake2 = Snake(init_pos=(3 * grid_width // 4, grid_height // 2), init_direction=LEFT, color=(0, 0, 255))
        self.food = self.spawn_food()

    def spawn_food(self):
        """
        Размещает еду в случайной ячейке, не занятой змейками.
        """
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake1.positions and pos not in self.snake2.positions:
                return pos

def get_state(snake, game):
    """
    Формирует состояние для агента в виде вектора из 6 элементов:
    [опасность прямо, опасность вправо, опасность влево, еда впереди, еда справа, еда слева]
    """
    head = snake.positions[0]
    # Опасность при движении прямо
    new_dir_straight = snake.direction
    new_head_straight = (head[0] + new_dir_straight[0], head[1] + new_dir_straight[1])
    danger_straight = 1.0 if is_danger(new_head_straight, game) else 0.0

    # Опасность при повороте направо
    new_dir_right = get_new_direction(snake.direction, 2)
    new_head_right = (head[0] + new_dir_right[0], head[1] + new_dir_right[1])
    danger_right = 1.0 if is_danger(new_head_right, game) else 0.0

    # Опасность при повороте налево
    new_dir_left = get_new_direction(snake.direction, 0)
    new_head_left = (head[0] + new_dir_left[0], head[1] + new_dir_left[1])
    danger_left = 1.0 if is_danger(new_head_left, game) else 0.0

    # Определяем положение еды относительно головы с учётом направления движения
    food = game.food
    food_ahead = 0.0
    food_right = 0.0
    food_left = 0.0

    if snake.direction == UP:
        if food[1] < head[1]:
            food_ahead = 1.0
        if food[0] > head[0]:
            food_right = 1.0
        if food[0] < head[0]:
            food_left = 1.0
    elif snake.direction == DOWN:
        if food[1] > head[1]:
            food_ahead = 1.0
        if food[0] < head[0]:
            food_right = 1.0
        if food[0] > head[0]:
            food_left = 1.0
    elif snake.direction == LEFT:
        if food[0] < head[0]:
            food_ahead = 1.0
        if food[1] < head[1]:
            food_right = 1.0
        if food[1] > head[1]:
            food_left = 1.0
    elif snake.direction == RIGHT:
        if food[0] > head[0]:
            food_ahead = 1.0
        if food[1] > head[1]:
            food_right = 1.0
        if food[1] < head[1]:
            food_left = 1.0

    state = [danger_straight, danger_right, danger_left, food_ahead, food_right, food_left]
    return torch.FloatTensor(state)

def check_collisions(game):
    """
    Проверяет столкновения для обеих змей и обновляет их флаг alive.
    """
    # Для змейки 1
    head1 = game.snake1.positions[0]
    if head1[0] < 0 or head1[0] >= game.grid_width or head1[1] < 0 or head1[1] >= game.grid_height:
        game.snake1.alive = False
    if head1 in game.snake1.positions[1:]:
        game.snake1.alive = False
    if head1 in game.snake2.positions:
        game.snake1.alive = False

    # Для змейки 2
    head2 = game.snake2.positions[0]
    if head2[0] < 0 or head2[0] >= game.grid_width or head2[1] < 0 or head2[1] >= game.grid_height:
        game.snake2.alive = False
    if head2 in game.snake2.positions[1:]:
        game.snake2.alive = False
    if head2 in game.snake1.positions:
        game.snake2.alive = False

def reset_snake(snake, init_pos, init_direction):
    """
    Сбрасывает змейку: задаёт начальную позицию, направление и делает её живой.
    """
    snake.positions = [init_pos]
    snake.direction = init_direction
    snake.alive = True
