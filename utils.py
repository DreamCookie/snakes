import random

# Параметры игры
BLOCK_SIZE = 20
GRID_WIDTH = 30    # ширина игрового поля (в блоках)
GRID_HEIGHT = 20   # высота игрового поля (в блоках)
FPS = 10           # частота кадров

# Определяем направления
UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)

def get_new_direction(current_direction, relative_move):
    """
    Вычисляет новое направление на основе текущего направления и относительного поворота.
    relative_move: 0 – налево, 1 – прямо, 2 – направо.
    """
    directions = [UP, RIGHT, DOWN, LEFT]
    idx = directions.index(current_direction)
    if relative_move == 0:
        new_idx = (idx - 1) % 4
    elif relative_move == 1:
        new_idx = idx
    elif relative_move == 2:
        new_idx = (idx + 1) % 4
    return directions[new_idx]

def is_danger(pos, game):
    """
    Проверяет является ли позиция опасной (выход за границы или столкновение со змейками).
    """
    x, y = pos
    if x < 0 or x >= game.grid_width or y < 0 or y >= game.grid_height:
        return True
    # Если в ячейке находится любая часть тела любой змейки – опасно
    if pos in game.snake1.positions or pos in game.snake2.positions:
        return True
    return False
