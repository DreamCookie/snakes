import pygame
import sys
import os
import torch
from utils import BLOCK_SIZE, GRID_WIDTH, GRID_HEIGHT, FPS, get_new_direction
from environment import SnakeGame, get_state, check_collisions, reset_snake
from agents import BetterAgent, WorseAgent

# Задаём ширину боковой панели
SIDE_PANEL_WIDTH = 100
# Общий размер окна: игровая область + две боковые панели
WINDOW_WIDTH = GRID_WIDTH * BLOCK_SIZE + 2 * SIDE_PANEL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Пути для сохранения весов агентов
AGENT1_WEIGHTS = "agent1_weights.pth"
AGENT2_WEIGHTS = "agent2_weights.pth"

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Две нейронки в змейке (RL)")
    clock = pygame.time.Clock()

    # Инициализация шрифта для отображения счёта
    font = pygame.font.SysFont("Arial", 24)

    # Инициализация игрового окружения и стартовых позиций змей
    game = SnakeGame(GRID_WIDTH, GRID_HEIGHT)
    snake1_start = (game.grid_width // 4, game.grid_height // 2)
    snake2_start = (3 * game.grid_width // 4, game.grid_height // 2)
    dir1_start = (1, 0)   # RIGHT
    dir2_start = (-1, 0)  # LEFT

    # Инициализация агентов
    agent1 = BetterAgent()  # агент с "умной" архитектурой
    agent2 = WorseAgent()   # агент с "хуже" настроенной сетью

    # Попытка загрузить сохранённые веса (если файлы существуют)
    if os.path.exists(AGENT1_WEIGHTS):
        agent1.load_model(AGENT1_WEIGHTS)
        print("Загружены веса для агента 1.")
    if os.path.exists(AGENT2_WEIGHTS):
        agent2.load_model(AGENT2_WEIGHTS)
        print("Загружены веса для агента 2.")

    while True:
        # Обработка событий (например, выход из игры)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Сохраняем веса перед завершением
                agent1.save_model(AGENT1_WEIGHTS)
                agent2.save_model(AGENT2_WEIGHTS)
                pygame.quit()
                sys.exit()

        # --- Агенты и выбор действий ---

        # Агент 1
        if game.snake1.alive:
            state1 = get_state(game.snake1, game)
            action1 = agent1.select_action(state1)
        else:
            state1 = get_state(game.snake1, game)
            action1 = None

        # Агент 2
        if game.snake2.alive:
            state2 = get_state(game.snake2, game)
            action2 = agent2.select_action(state2)
        else:
            state2 = get_state(game.snake2, game)
            action2 = None

        new_dir1 = get_new_direction(game.snake1.direction, action1) if action1 is not None else game.snake1.direction
        new_dir2 = get_new_direction(game.snake2.direction, action2) if action2 is not None else game.snake2.direction

        head1_next = (game.snake1.positions[0][0] + new_dir1[0], game.snake1.positions[0][1] + new_dir1[1])
        head2_next = (game.snake2.positions[0][0] + new_dir2[0], game.snake2.positions[0][1] + new_dir2[1])
        food_eaten1 = (head1_next == game.food)
        food_eaten2 = (head2_next == game.food)
        # Обновляем счет при поедании еды
        if food_eaten1:
            game.snake1.score += 1
        if food_eaten2:
            game.snake2.score += 1

        if game.snake1.alive:
            game.snake1.move(new_dir1, food_eaten1)
        if game.snake2.alive:
            game.snake2.move(new_dir2, food_eaten2)

        check_collisions(game)

        # --- Обработка наград и обучение агента 1 ---
        if not game.snake1.alive:
            reward1 = -10.0
            done1 = True
            reset_snake(game.snake1, snake1_start, dir1_start)
        elif food_eaten1:
            reward1 = 10.0
            done1 = False
        else:
            reward1 = -0.1
            done1 = False

        new_state1 = get_state(game.snake1, game)
        if action1 is not None:
            agent1.store_transition(state1, action1, reward1, new_state1, done1)
            agent1.train_model()

        # --- Обработка наград и обучение агента 2 ---
        if not game.snake2.alive:
            reward2 = -10.0
            done2 = True
            reset_snake(game.snake2, snake2_start, dir2_start)
        elif food_eaten2:
            reward2 = 10.0
            done2 = False
        else:
            reward2 = -0.1
            done2 = False

        new_state2 = get_state(game.snake2, game)
        if action2 is not None:
            agent2.store_transition(state2, action2, reward2, new_state2, done2)
            agent2.train_model()

        # Если еда съедена, создаём новую
        if food_eaten1 or food_eaten2:
            game.food = game.spawn_food()

        # --- Отрисовка ---

        # Заполняем фон чёрным
        screen.fill((0, 0, 0))

        # Отрисовка боковых панелей (фон панелей — тёмно-серый)
        left_panel_rect = pygame.Rect(0, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, (50, 50, 50), left_panel_rect)
        right_panel_rect = pygame.Rect(SIDE_PANEL_WIDTH + GRID_WIDTH * BLOCK_SIZE, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(screen, (50, 50, 50), right_panel_rect)

        # Отрисовка текста со счётом
        score_text1 = font.render(f"Snake 1: {game.snake1.score}", True, (255, 255, 255))
        score_text2 = font.render(f"Snake 2: {game.snake2.score}", True, (255, 255, 255))
        screen.blit(score_text1, (10, 10))  # в левой панели
        screen.blit(score_text2, (SIDE_PANEL_WIDTH + GRID_WIDTH * BLOCK_SIZE + 10, 10))  # в правой панели

        # Смещение для отрисовки игровой области (отступ равен ширине левой панели)
        board_offset_x = SIDE_PANEL_WIDTH

        # Отрисовка еды (с учётом смещения)
        food_rect = pygame.Rect(board_offset_x + game.food[0] * BLOCK_SIZE,
                                game.food[1] * BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)

        # Отрисовка змейки 1: каждый квадратик отрисовывается с обводкой
        for pos in game.snake1.positions:
            rect = pygame.Rect(board_offset_x + pos[0] * BLOCK_SIZE,
                               pos[1] * BLOCK_SIZE,
                               BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, game.snake1.color, rect)          # заливка
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)                # контур (обводка)

        # Отрисовка змейки 2 с обводкой
        for pos in game.snake2.positions:
            rect = pygame.Rect(board_offset_x + pos[0] * BLOCK_SIZE,
                               pos[1] * BLOCK_SIZE,
                               BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, game.snake2.color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == '__main__':
    main()
